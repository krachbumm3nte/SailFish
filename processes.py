import sys

import SimpleITK as sitk
import numpy as np


class Process:
    """
    superclass for all processes to be defined for the pipeline.
    to add a process, simply create a class in this file that inherits its functionality from this class.
    Attributes:
        description (dict): the python dict generated from this process' corresponding JSON-object
        name (string): Same as the classname, used for logging purposes
        show_output (bool, optional): If True, the output image (or chunk) of the calculation will be displayed.
            Defaults to False.
        chunking_optimized (bool, optional): If False, an alert will be printed to the console when a process is
            calculated using chunking. Defaults to True.
        master (ProcessHandler): a reference this instances creating Processhandler.
        logger (Logger): encapsulates the output of basic user information during computation
    """

    def __init__(self, description, chunking_optimized=True):
        self.description = description
        self.name = self.get_attr_by_name('type')  # definition of a required key
        self.show_output = self.get_attr_by_name('show_output', False)  # definition of an optional key
        self.chunking_optimized = chunking_optimized
        self.master = None
        self.logger = None

    def set_master(self, handler):
        self.master = handler

    def set_logger(self, logger):
        self.logger = logger

    def execute(self, input_image, enable_chunking=False):
        self.logger.log_started(self.name)
        if enable_chunking:
            if not self.chunking_optimized:
                self.logger.log_warning(f"{self.name} is not optimized for chunking, image faults may occur!")
            result = self.calculate_chunk(input_image)
        else:
            result = self.calculate(input_image)

        self.logger.log_completed(self.name)

        if self.show_output:
            self.master.io_handler.show_3D_image(result)
        return result

    def calculate(self, input_image):
        return input_image

    def calculate_chunk(self, input_image):
        return self.calculate(input_image)

    def get_attr_by_name(self, attr_name, default=None):
        """
        retrieves a property from the JSON-object that created this process. Is also used to define the Keys that are
        required to instantiate a process
        :param attr_name:
        :param default:
        :return:
        """
        if attr_name in self.description.keys():
            return self.description[attr_name]
        elif default is not None:
            return default
        else:
            print(f"could not instantiate process {self.name}: Atribute {attr_name} was not found in input")
            sys.exit()


# begin defining custom processes from here

class SmoothingGaussianFilter(Process):

    def __init__(self, description):
        super().__init__(description)
        self.sigma = self.get_attr_by_name('sigma')

    def calculate(self, input_image):
        pixel_id = input_image.GetPixelID()

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(self.sigma)
        img_temp = gaussian.Execute(input_image)

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixel_id)
        return caster.Execute(img_temp)


class ConnectedThresholding(Process):
    """
    applies a connected Thresholding (region growth) on an image or chunk. If chunking is enabled, the seeds for every
    chunk following the first one will be calculated from the last slice of the previous one.
    """

    def __init__(self, description):
        super().__init__(description)
        self.lower = self.get_attr_by_name('lower')
        self.upper = self.get_attr_by_name('upper')
        self.seeds = self.get_attr_by_name('seeds')
        self.replacevalue = self.get_attr_by_name('replacevalue', 1)

    def calculate(self, input_image):
        self.seeds = self.parse_seeds(self.seeds, self.master.rescaling_factor)
        return sitk.ConnectedThreshold(image1=input_image,
                                       seedList=self.seeds,
                                       lower=self.lower,
                                       upper=self.upper,
                                       replaceValue=self.replacevalue)

    def calculate_chunk(self, input_image):
        if self.master.current_chunk == 0:
            self.seeds = self.parse_seeds(self.seeds, self.master.rescaling_factor)
        result = sitk.ConnectedThreshold(image1=input_image,
                                         seedList=self.seeds,
                                         lower=self.lower,
                                         upper=self.upper,
                                         replaceValue=self.replacevalue)

        self.update_seeds(sitk.GetArrayFromImage(result[:, :, -1]))
        self.logger.log_timestamp(f"{len(self.seeds)} seeds defined for the next chunk")
        return result

    def parse_seeds(self, seeds, scale):
        return [(int(x / scale), int(y / scale), int(z / scale)) for (x, y, z) in seeds]

    def update_seeds(self, last_slice):
        seeds = []
        len_x, len_y = last_slice.shape
        for x in range(len_x):
            for y in range(len_y):
                if last_slice[x, y] == self.replacevalue:
                    seeds.append((y, x, 0))
        self.seeds = seeds


class InvertIntensity(Process):

    def __init__(self, description):
        super().__init__(description, False)

    def calculate(self, input_image):
        return sitk.InvertIntensity(sitk.RescaleIntensity(input_image))


class MaskImage(Process):

    def __init__(self, description):
        super().__init__(description)
        self.offset = self.get_attr_by_name('offset', 0)
        self.mask_location = self.get_attr_by_name('mask_location')
        self.mask_downscale = self.get_attr_by_name('mask_downscale', 1)

    def calculate(self, input_image):
        start_index, end_index = self.master.current_indices
        mask = self.master.io_handler.load_from_file(self.mask_location, self.mask_downscale,
                                                     start_index - self.offset, end_index - self.offset)
        return sitk.Mask(input_image, mask)


class BinaryErosion(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.get_attr_by_name('radius')

    def calculate(self, input_image):
        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetBackgroundValue(0)
        erode_filter.SetForegroundValue(255)
        erode_filter.SetKernelRadius(self.radius)
        return erode_filter.Execute(input_image)


class BinaryDilation(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.get_attr_by_name('radius')

    def calculate(self, input_image):
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetBackgroundValue(0)
        dilate_filter.SetForegroundValue(255)
        dilate_filter.SetKernelRadius(self.radius)
        return dilate_filter.Execute(input_image)


class MeanFilter(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.get_attr_by_name('radius')

    def calculate(self, input_image):
        image_filter = sitk.MeanImageFilter()
        image_filter.SetRadius(int(self.radius))
        return image_filter.Execute(input_image)


class OtsuThresholding(Process):
    def calculate(self, input_image):
        return sitk.OtsuThreshold(input_image)


class RescaleToOriginal(Process):
    """
    rescales the image or chunk to its original size
    """

    def calculate(self, input_image):
        start, end = self.master.current_indices
        data_shape_orig = self.master.json_interpreter.input_shape_unscaled
        target_shape = (end - start, data_shape_orig[1], data_shape_orig[2] )
        return self.rescale(input_image, target_shape)

    def calculate_chunk(self, input_image):
        shape_orig = self.master.json_interpreter.input_shape_unscaled
        current_chunk = self.master.current_indices
        target_shape = (current_chunk[1] - current_chunk[0], shape_orig[1], shape_orig[2])
        return self.rescale(input_image, target_shape)

    def rescale(self, input_image, target_shape):
        self.logger.log_timestamp(f"rescaling to target shape {target_shape}")
        in_array = sitk.GetArrayFromImage(input_image)
        rescaling_factor = self.master.rescaling_factor
        rescale_volume = (rescaling_factor, rescaling_factor, rescaling_factor)
        self.logger.log_timestamp(f"applying kronecker product with shape: {rescale_volume}")
        out_array = np.kron(in_array, np.ones(rescale_volume, in_array.dtype))

        for i in range(3):
            if out_array.shape[i] < target_shape[i]:
                self.logger.log_error("invalid rescaling factor: output dimension is smaller than target dimension")
        out_array = out_array[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
        out_image = sitk.GetImageFromArray(out_array)
        return out_image


class AppendImages(Process):
    """
    Appends a list of images specified by their location to an image or chunk. The original input will always be the
    first image along the X-Axis. All images need to have the exact same shape along their Y- and Z-Axes.
    Can be used to reassemble images that needed to be split up due to hardware limitations
    """

    def __init__(self, description):
        super().__init__(description)
        self.image_locations = self.get_attr_by_name("images")

    def calculate(self, input_image):
        result = sitk.GetArrayFromImage(input_image)
        for location in self.image_locations:
            self.logger.log_timestamp(f"processing: {location}")
            current_image_data = self.master.io_handler.load_array_from_file(location, 1)
            if result.shape[1:] != current_image_data.shape[1:]:
                self.logger.log_error(f"cannot append image because of incorrect dimension."
                                      f"\nexisting image shape: {result.shape}, new image shape: {current_image_data.shape}"
                                      f"\nimage location: {location}")
            result = np.concatenate((result, current_image_data), axis=0)

        self.logger.log_timestamp(f"Assembly finished, resulting size: {result.shape}")
        return sitk.GetImageFromArray(result)


class Thresholding(Process):

    def __init__(self, description):
        super().__init__(description)
        self.lower = self.get_attr_by_name("lower")
        self.upper = self.get_attr_by_name("upper")

    def calculate(self, input_image):
        return sitk.Threshold(input_image, self.lower, self.upper, 0)


class GrayScaleDilation(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.get_attr_by_name('radius')

    def calculate(self, input_image):
        return sitk.GrayscaleDilate(input_image, self.radius)


class GrayScaleErosion(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.get_attr_by_name('radius')

    def calculate(self, input_image):
        return sitk.GrayscaleErode(input_image, self.radius)


class LogicalOr(Process):

    def __init__(self, description):
        super().__init__(description)
        self.image_location = self.get_attr_by_name('image_location')

    def calculate(self, input_image):
        second_image = self.master.io_handler.load_from_file(self.image_location, self.master.rescaling_factor,
                                                             self.master.current_indices[0],
                                                             self.master.current_indices[1])

        return sitk.Or(input_image, second_image)


class LogicalAnd(Process):

    def __init__(self, description):
        super().__init__(description)
        self.image_location = self.get_attr_by_name('image_location')

    def calculate(self, input_image):
        second_image = self.master.io_handler.load_from_file(self.image_location, self.master.rescaling_factor,
                                                             self.master.current_indices[0],
                                                             self.master.current_indices[1])

        return sitk.And(input_image, second_image)


"""   

class Foo(Process):

def __init__(self, description):
   super().__init__(description)

def calculate(self, input_image):
   pass

class Process(Process):

def __init__(self, description):
   super().__init__(description)

def calculate(self, input_image):
   pass
"""
