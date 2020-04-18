from PIL import ImageEnhance
from PIL import Image as pil_image
import numpy as np
import random
from skimage.util import random_noise
from mlebe.training.utils import data_normalization

def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
        dtype: Dtype to use.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))

def apply_brightness_shift(x, brightness):
    """Performs a brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """
    if ImageEnhance is None:
        raise ImportError('Using brightness shifts requires PIL. '
                          'Install PIL or Pillow.')
    x = array_to_img(x)
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = img_to_array(x)
    return x


def random_brightness(x, brightness_range):
    """Performs a random brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness_range: Tuple of floats; brightness range.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """
    if len(brightness_range) != 2:
        raise ValueError(
            '`brightness_range should be tuple or list of two floats. '
            'Received: %s' % (brightness_range,))

    u = np.random.uniform(brightness_range[0], brightness_range[1])
    return apply_brightness_shift(x, u)

def gauss(x, a, m, s):
    return np.sqrt(a) * np.exp(-(x - m) ** 2 / (2 * s ** 2))

def bias(x):

    return np.exp(np.multiply(a, np.multiply(x, y)))

def gaussian_bias(img, mask, var_range):
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    a = np.where(mask == 0)
    listofCoord = list(zip(a[0], a[1]))
    coord = listofCoord[random.randint(0, len(listofCoord)-1)]
    varx = random.randint(var_range[0], var_range[1] + 1)
    meanx = coord[0]
    vary = random.randint(var_range[0], var_range[1] + 1)
    meany = coord[1]
    maxx = float(random.randint(255, 500))/255
    maxy= float(random.randint(255, 500))/255
    gaus2d = gauss(x, maxx, meanx, varx)*gauss(y, maxy, meany, vary)
    if np.max(gaus2d) == 0:
        augmented = img
    elif np.max(img) == 0:
        augmented = gaus2d
    else:
        # augmented = np.multiply(img, gaus2d)
        augmented = img + gaus2d
        augmented = augmented / np.max(augmented)
    return augmented

class Augment(object):
    def __init__(self, brightness_range = None, noise_var_range = None, bias_var_range = None, type = 'scan', save_dir = ''):
        """

        :param brightness_range:
        :param noise_var_range:
        :param bias_var_range:
        :param type:
        :param save_dir:
        """
        self.brightness_range = brightness_range
        self.noise_var_range = noise_var_range
        self.bias_var_range = bias_var_range
        self.type = type
        self.scan_count = 0
        self.scan_list = []
        self.save_dir = save_dir
        self.mask_count = 0
        self.mask_list = []


    def __call__(self, input_data):
        x = input_data
        if self.type == 'scan':
            x = random_brightness(x, self.brightness_range) * (1. / 255)
            x = np.squeeze(x)
            # mask = np.squeeze(mask)
            # if random.random() < 0.01:
            #     x = gaussian_bias(x, mask, self.bias_var_range)
            var = random.uniform(self.noise_var_range[0], self.noise_var_range[1])
            x = random_noise(x, mode='gaussian', var=var)
            x = np.expand_dims(data_normalization(x), -1)
            if self.scan_count < 51:
                self.scan_list.append(x)
            if self.scan_count == 51:
                np.save(self.save_dir + 'x_train_augmented', np.array(self.scan_list))
            self.scan_count += 1
            print('img count: ', self.scan_count)
            return x / np.max(x).astype('float32')
        elif self.type == 'mask':
            if self.mask_count < 51:
                self.mask_list.append(x)
            if self.mask_count == 51:
                np.save(self.save_dir + 'x_train_augmented', np.array(self.mask_list))
            self.mask_count += 1
            print('img count: ', self.scan_count)
            return np.where(x > 0.5, 1, 0)

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir


def augment(x, mask, brightness_range, noise_var_range, bias_var_range):
    """
    :param x: single image
    :param mask:
    :param brightness_range:
    :param noise_var_range:
    :param bias_var_range:
    :return:
    """
    x = random_brightness(x, brightness_range)*(1./255)
    x = np.squeeze(x)
    mask = np.squeeze(mask)
    if random.random() < 0.01:
        x = gaussian_bias(x, mask , bias_var_range)
    var = random.uniform(noise_var_range[0], noise_var_range[1])
    x = random_noise(x, mode = 'gaussian', var = var)
    x = np.expand_dims(data_normalization(x), -1)
    return x/np.max(x)