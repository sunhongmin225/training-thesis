import numpy as np
import torch
import pytest
from numpy.lib import NumpyVersion
import scipy
from scipy import ndimage as ndi

import filters

# from .._geometric import (SimilarityTransform, AffineTransform,
#                          ProjectiveTransform)
# from .._warps_cy import _warp_fast
# from ...measure import block_reduce

from msh_utils import (get_bound_method_class, safe_as_int, warn,
                             _to_ndimage_mode,
                             channel_as_last_axis,
                             deprecate_multichannel_kwarg)

from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)
from scipy.ndimage import map_coordinates

from skimage._shared.testing import expected_warnings, test_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color.colorconv import rgb2gray
from skimage.data import checkerboard, astronaut
from skimage.draw.draw import circle_perimeter_aa
from skimage.feature.peak import peak_local_max
from skimage.transform._warps import (_stackcopy,
                                      _linear_polar_mapping,
                                      _log_polar_mapping, warp,
                                      warp_coords, rotate,
                                      rescale, warp_polar, swirl,
                                      downscale_local_mean,
                                      resize_local_mean)
from skimage.transform._geometric import (AffineTransform,
                                          ProjectiveTransform,
                                          SimilarityTransform)
from skimage.util.dtype import img_as_float, _convert
from absl import logging

np.random.seed(0)


def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    # print("convert_to_float image.dtype: {}".format(image.dtype))
    if image.dtype == torch.float16:
        return image.astype(torch.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from msh_dtype import img_as_float
        image = img_as_float(image)
    return image


def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
             "with bool data type. Please set order to 0 or explicitely "
             "cast input image to another data type.")

    return order


def _preprocess_resize_output_shape(image, output_shape):
    """Validate resize output shape according to input image.

    Parameters
    ----------
    image: ndarray
        Image to be resized.
    output_shape: tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved.

    Returns
    -------
    image: ndarray
        The input image, but with additional singleton dimensions appended in
        the case where ``len(output_shape) > input.ndim``.
    output_shape: tuple
        The output image converted to tuple.

    Raises
    ------
    ValueError:
        If output_shape length is smaller than the image number of
        dimensions

    Notes
    -----
    The input image is reshaped if its number of dimensions is not
    equal to output_shape_length.

    """
    output_ndim = len(output_shape)
    input_shape = image.shape
    # print("input_shape: {}".format(input_shape))
    # print("output_ndim: {}, image.ndim: {}".format(output_ndim, image.ndim))
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape += (1, ) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1], )
    elif output_ndim < image.ndim:
        raise ValueError("output_shape length cannot be smaller than the "
                         "image number of dimensions")

    return image, output_shape


def _clip_warp_output(input_image, output_image, mode, cval, clip):
    """Clip output image to range of values of input image.

    Note that this function modifies the values of `output_image` in-place
    and it is only modified if ``clip=True``.

    Parameters
    ----------
    input_image : ndarray
        Input image.
    output_image : ndarray
        Output image, which is modified in-place.

    Other parameters
    ----------------
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.

    """

    # print("input_image: {}".format(input_image))
    # print("output_image: {}".format(output_image))
    # print("mode: {}".format(mode))
    # print("cval: {}".format(cval))
    # print("clip: {}".format(clip))

    if clip:
        min_val = input_image.min()
        max_val = input_image.max()

        preserve_cval = (mode == 'constant' and not
                         (min_val <= cval <= max_val))

        if preserve_cval:
            cval_mask = output_image == cval

        np.clip(output_image, min_val, max_val, out=output_image)

        if preserve_cval:
            output_image[cval_mask] = cval


def resize(image, output_shape, order=None, mode='reflect', cval=0, clip=True,
           preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None):
    """Resize image to match a certain size.

    Performs interpolation to up-size or down-size N-dimensional images. Note
    that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For downsampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to downsampling. It is crucial to filter when downsampling
        the image to avoid aliasing artifacts. If not specified, it is set to
        True when downsampling an image whose data type is not bool.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering used when anti-aliasing.
        By default, this value is chosen as (s - 1) / 2 where s is the
        downsampling factor, where s > 1. For the up-size case, s < 1, no
        anti-aliasing is performed prior to rescaling.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import resize
    >>> image = data.camera()
    >>> resize(image, (100, 100)).shape
    (100, 100)

    """
    # print("\nimage: \n{}\n output_shape: {}\n".format(image, output_shape))
    logging.info("msh::start _preprocess_resize_output_shape")
    image, output_shape = _preprocess_resize_output_shape(image, output_shape)
    logging.info("msh::done _preprocess_resize_output_shape")
    # print("\nimage after: \n{}\n output_shape after: {}\n".format(image, output_shape))

    input_shape = image.shape
    input_type = image.dtype

    # print("input_type: {}".format(input_type))
    # print("input_shape after: \n{}\n output_shape after: {}\n".format(input_shape, output_shape))

    if input_type == torch.float16:
        image = image.astype(torch.float32)

    if anti_aliasing is None:
        anti_aliasing = (not input_type == bool and
                         any(x < y for x, y in zip(output_shape, input_shape)))

    # print("anti_aliasing: {}".format(anti_aliasing))

    if input_type == bool and anti_aliasing:
        raise ValueError("anti_aliasing must be False for boolean images")

    factors = np.divide(input_shape, output_shape)
    # print("factors: {}".format(factors))

    logging.info("msh::start _validate_interpolation_order")
    order = _validate_interpolation_order(input_type, order)
    logging.info("msh::done _validate_interpolation_order")

    # print("\nimage: \n{}".format(image))

    # TODO: msh
    if order > 0:
        logging.info("msh::start convert_to_float")
        image = convert_to_float(image, preserve_range)
        logging.info("msh::done convert_to_float")

    # print("\nimage after: \n{}".format(image))

    # Save input value range for clip
    img_bounds = np.array([image.min(), image.max()]) if clip else None

    # img_bounds = torch.from_numpy(img_bounds)
    # print("img_bounds after: \n{}".format(img_bounds))

    # TODO: msh
    # Translate modes used by np.pad to those used by scipy.ndimage
    ndi_mode = _to_ndimage_mode(mode)
    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        else:
            anti_aliasing_sigma = \
                np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
            if np.any(anti_aliasing_sigma < 0):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")
            elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                warn("Anti-aliasing standard deviation greater than zero but "
                     "not down-sampling along all axes")
        logging.info("msh::start gaussian_filter")
        image = filters.gaussian_filter(image, anti_aliasing_sigma,
                                    cval=cval, mode=ndi_mode)
        logging.info("msh::done gaussian_filter")


    if NumpyVersion(scipy.__version__) >= '1.6.0':
        # print("AA")
        # The grid_mode kwarg was introduced in SciPy 1.6.0
        zoom_factors = [1 / f for f in factors]
        # print("msh::zoom_factors: {}".format(zoom_factors))
        logging.info("msh::start ndi.zoom")
        out = ndi.zoom(image, zoom_factors, order=order, mode=ndi_mode,
                       cval=cval, grid_mode=True)
        logging.info("msh::done ndi.zoom")

    # TODO: Remove the fallback code below once SciPy >= 1.6.0 is required.

    # 2-dimensional interpolation
    elif len(output_shape) == 2 or (len(output_shape) == 3 and
                                    output_shape[2] == input_shape[2]):
        # print("BB")
        rows = output_shape[0]
        cols = output_shape[1]
        input_rows = input_shape[0]
        input_cols = input_shape[1]
        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
                                                 input_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        # Make sure the transform is exactly metric, to ensure fast warping.
        tform.params[2] = (0, 0, 1)
        tform.params[0, 1] = 0
        tform.params[1, 0] = 0

        # clip outside of warp to clip w.r.t input values, not filtered values.
        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=False,
                   preserve_range=preserve_range)

    else:  # n-dimensional interpolation
        # print("CC")

        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        coord_map = np.array(np.meshgrid(*coord_arrays,
                                         sparse=False,
                                         indexing='ij'))

        out = ndi.map_coordinates(image, coord_map, order=order,
                                  mode=ndi_mode, cval=cval)

    logging.info("msh::start _clip_warp_output")
    _clip_warp_output(img_bounds, out, mode, cval, clip)
    logging.info("msh::done _clip_warp_output")

    return out


def test_resize2d():
    x = np.zeros((3, 6), dtype=np.double)
    x[1, 1] = 1
    print(x)
    t = torch.from_numpy(x)
    print(t)
    # resized = resize(x, (5, 8))
    resized = resize(t, (5, 8))
    print(resized)
    # ref = np.zeros((10, 10))
    # ref[2:4, 2:4] = 1
    # assert_array_almost_equal(resized, ref)



# test_resize2d()
