import math
import numbers
import torch
import filters
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
import time
import numpy as np

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        print("input.shape: {}, self.weight.shape: {}, self.groups: {}".format(input.shape, self.weight.shape, self.groups))
        input = input.type(torch.double)
        self.weight = self.weight.type(torch.double)
        print("input: \n{}, \nself.weight: \n{}".format(input, self.weight))
        return self.conv(input, weight=self.weight, groups=self.groups)


# image = np.random.rand(9960, 2)
# row, col = image.shape
# image = torch.from_numpy(image)
# print("image: \n{}".format(image))
# output = np.array([2, 2])
# factors = np.divide(image.shape, output.shape)
# anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
# print("anti_aliasing_sigma: {}".format(anti_aliasing_sigma))
# kernel_size = (2 * row - 1, 2 * col - 1)
# smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=(anti_aliasing_sigma[0] + 1, anti_aliasing_sigma[1] + 1), dim=2)
# image = torch.reshape(image, (1, row, col))
# image = torch.unsqueeze(image, 0)
# print("image.shape after unsqueeze: {}".format(image.shape))
# torch_start = time.time()
# input_pad = F.pad(image, (col - 1, col - 1, row - 1, row - 1), mode='reflect')
# output = smoothing(input_pad)
# torch_end = time.time()
# output = torch.squeeze(output)
# print("output.shape after squeeze: {}".format(output.shape))
# print("output: \n{}".format(output))


# cval = 0
# ndi_mode = "mirror"
# image = torch.squeeze(image)
# print("image.shape after squeeze: {}".format(image.shape))
# start = time.time()
# output = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
# # output = filters.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
# end = time.time()
# print("output.shape: {}".format(output.shape))
# print("output: \n{}".format(output))

# print("torch total: {}".format(torch_end - torch_start))
# print("ndi total: {}".format(end - start))
