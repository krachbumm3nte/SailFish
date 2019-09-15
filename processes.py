import sys
import io_utils
import logger
import process_handler
import utils
import SimpleITK as sitk
import settings
import numpy as np
import io_utils


class Process:
    """
    superclass for all processes to be defined for the pipeline.
    to add a process, simply create a class in this file that inherits its functionality from this class.
    """

    def __init__(self, description, chunking_optimized=True):
        self.description = description  # TODO: remove this?
        self.name = self.retreive_attribute('type')
        self.save_to_disk = self.retreive_attribute('save_flag', False)
        self.chunking_optimized = chunking_optimized
        self.handler: process_handler.ProcessHandler = None

    def set_handler(self, handler):
        self.handler = handler

    def execute(self, input_image, enable_chunking=False):
        logger.log_started(self.name)
        if enable_chunking:
            if not self.chunking_optimized:
                logger.log_warning(f"{self.name} is not optimized for chunking, image faults may occur")
            result = self.calculate_chunk(input_image)
        else:
            result = self.calculate(input_image)

        logger.log_completed(self.name)
        return result


    def calculate(self, input_image):
        return input_image

    def calculate_chunk(self, input_image):
        return self.calculate(input_image)


    def retreive_attribute(self, attr_name, default=None):
        if attr_name in self.description.keys():
            return self.description[attr_name]
        elif default is not None:
            return default
        else:
            # TODO: proper exception handling
            raise Exception(f"{attr_name} required for process")


# begin defining custom processes from here

class SmoothingGaussianFilter(Process):

    def __init__(self, description):
        super().__init__(description)
        self.sigma = self.retreive_attribute('sigma')

    def calculate(self, input_image):
        pixel_id = input_image.GetPixelID()

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(self.sigma)
        img_temp = gaussian.Execute(input_image)

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixel_id)
        return caster.Execute(img_temp)


class ConnectedThresholding(Process):

    def __init__(self, description):
        super().__init__(description)
        self.seeds = utils.parse_seeds(self.retreive_attribute('seeds'), settings.rescaling_factor)
        self.lower = self.retreive_attribute('lower')
        self.upper = self.retreive_attribute('upper')
        self.replacevalue = self.retreive_attribute('replacevalue', 1)

    def calculate(self, input_image):
        return sitk.ConnectedThreshold(image1=input_image,
                                       seedList=self.seeds,
                                       lower=self.lower,
                                       upper=self.upper,
                                       replaceValue=self.replacevalue)

    def calculate_chunk(self, input_image):
        result = self.calculate(input_image)

        # dilate last slice of current image by one pixel to ensure proper continuation on the next chunk
        last_slice = sitk.BinaryDilate(result[:, :, -1], 1)

        self.update_seeds(sitk.GetArrayFromImage(last_slice))

        return result

    def update_seeds(self, array):
        seeds = []
        len_x, len_y = array.shape
        for x in range(len_x):
            for y in range(len_y):
                if array[x, y] == self.replacevalue:
                    seeds.append((y, x, 0))
        self.seeds = seeds




class SliceWiseMasking(Process):

    def __init__(self, description):
        super().__init__(description)
        self.lower = self.retreive_attribute('lower')
        self.upper = self.retreive_attribute('upper')
        self.replacevalue = self.retreive_attribute('replacevalue')

    def calculate(self, input_image: sitk.Image):
        length, height, width= input_image.GetDepth(), input_image.GetHeight(), input_image.GetWidth()
        result = np.ndarray(shape=(length, height, width))
        print(result.shape)
        seeds = [(0, 0), (0, width-1), (height - 1, 0), (height - 1, width -1)]
        for i in range(length):
            result[i,:,:] = sitk.GetArrayFromImage(sitk.ConnectedThreshold(image1=input_image[:,:,i],
                                                   seedList=seeds,
                                                   lower=self.lower,
                                                   upper=self.upper,
                                                   replaceValue=self.replacevalue))
        print("growing done")
        print(result.shape)
        result = sitk.GetImageFromArray(result)
        print("what")
        if (length, height, width) == (result.GetDepth(), result.GetHeight(), result.GetWidth()):
            print("yay")
        else:
            sys.exit()
        return result


class SliceWiseMasking_2(Process):

    def __init__(self, description):
        super().__init__(description)
        self.lower = self.retreive_attribute('lower')
        self.upper = self.retreive_attribute('upper')
        self.replacevalue = self.retreive_attribute('replacevalue')

    def calculate(self, input_image: sitk.Image):
        length, height, width= input_image.GetDepth(), input_image.GetHeight(), input_image.GetWidth()
        seeds = [(0, 0), (0, width-1), (height - 1, 0), (height - 1, width -1)]
        for i in range(length):
            temp = sitk.ConnectedThreshold(image1=input_image[:,:,i],
                                           seedList=seeds,
                                           lower=self.lower,
                                           upper=self.upper,
                                           replaceValue=self.replacevalue)
            temp = sitk.JoinSeries(temp)
            input_image = sitk.Paste(input_image, temp, temp.GetSize(), destinationIndex=[0,0,i])

        return input_image


class InvertIntensity(Process):

    def __init__(self, description):
        super().__init__(description)

    def calculate(self, input_image):
        return sitk.InvertIntensity(sitk.RescaleIntensity(input_image))

    def calculate_chunk(self, input_image):
        logger.log_warning("Inversion is not optimized for chunking, image faults may occur")
        return self.calculate(input_image)


class MaskImage(Process):

    def __init__(self, description):
        super().__init__(description)
        self.mask_location = self.retreive_attribute('mask_location')
        self.mask_downscale = self.retreive_attribute('mask_downscale', 1)

    def calculate(self, input_image):
        start_index, end_index = self.handler.current_indices
        mask = io_utils.load_from_file(self.mask_location, 1, start_index-2400, end_index-2400)
        print(mask.GetSize(), mask.GetPixelIDTypeAsString())
        print(input_image.GetSize(), input_image.GetPixelIDTypeAsString())
        return sitk.Mask(input_image, mask)

    def calculate_chunk(self, input_image):
        return self.calculate(input_image)


class BinaryErosion(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.retreive_attribute('radius')

    def calculate(self, input_image):
        return sitk.BinaryErode(input_image, self.radius)


class BinaryDilation(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.retreive_attribute('radius'
                                              '')

    def calculate(self, input_image):
        return sitk.BinaryDilate(input_image, self.radius)


class MeanFilter(Process):

    def __init__(self, description):
        super().__init__(description)
        self.radius = self.retreive_attribute('radius')

    def calculate(self, input_image):
        image_filter = sitk.MeanImageFilter()
        image_filter.SetRadius(int(self.radius))
        return image_filter.Execute(input_image)

      
class ReassembleChunks(Process):

  def __init__(self, description):
      super().__init__(description)

  def calculate(self, input_image):
      pass

      
class OtsuThresholding(Process):

    def calculate(self, input_image):
        return sitk.OtsuThreshold(input_image)


class RescaleToOriginal(Process):

    def calculate(self, input_image):
        return self.rescale(input_image, self.handler.current_shape_unscaled)

    def calculate_chunk(self, input_image):
        in_shape = self.handler.current_shape_unscaled
        curr_chunk = self.handler.current_indices
        target_shape = (curr_chunk[1]-curr_chunk[0], in_shape[1], in_shape[2])
        return self.rescale(input_image, target_shape)


    def rescale(self, input_image, target_shape):
        logger.log_info(f"rescaling to target shape {target_shape}")
        in_array = sitk.GetArrayFromImage(input_image)
        rescaling_factor = self.handler.rescaling_factor
        rescale_volume = (rescaling_factor, rescaling_factor, rescaling_factor)
        print(f"applying kronecker product with shape: {rescale_volume}")
        out_array = np.kron(in_array, np.ones(rescale_volume, in_array.dtype))

        for i in range(3):
            if out_array.shape[i] < target_shape[i]:
                logger.log_error("invalid rescaling factor: output dimension is smaller than target dimension")
        print(out_array.shape)
        out_array = out_array[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
        out_image = sitk.GetImageFromArray(out_array)
        print(in_array.shape, out_array.shape)
        return out_image



class AppendImages(Process):

    def __init__(self, description):
        super().__init__(description)
        self.image_locations = self.retreive_attribute("images")

    def calculate(self, input_image):
        result = sitk.GetArrayFromImage(input_image)
        for location in self.image_locations:
            logger.log_timestamp(f"processing: {location}")
            current_image_data = io_utils.load_array_from_file(location)
            print(current_image_data.shape)
            result = np.concatenate((result, current_image_data), axis=0)

        logger.log_info(f"Assembly finished, resulting size: {result.shape}")
        return sitk.GetImageFromArray(result)

"""

      
      class Process(Process):

  def __init__(self, description):
      super().__init__(description)

  def calculate(self, input_image):
      pass
"""