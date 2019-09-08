import sys

import io_utils
import logger
import utils
import SimpleITK as sitk
import settings
import numpy as np


class Process:
    """
    superclass for all processes to be defined for the pipeline.
    to add a process, simply create a class in this file that inherits its functionality from this class.
    """

    def __init__(self, description, chunking_optimized=True):
        self.description = description  # TODO: remove this?
        self.name = self.retreive_attribute('type')
        self.save_to_disk = self.retreive_attribute('save_flag', False)
        self.input = self.retreive_attribute('input', -1)
        self.chunking_optimized = chunking_optimized


    def execute(self, input_image, enable_chunking=False):
        # TODO: chunking aus settings abfragen, statt parameter durchzureichen
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
        last_slice = sitk.GetArrayFromImage(result)[-1]
        self.update_seeds(last_slice)
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
        start_index, end_index = settings.current_chunk[1:]
        mask = io_utils.load_from_file(self.mask_location, start_index, end_index)
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
      """
      
      class Process(Process):

  def __init__(self, description):
      super().__init__(description)

  def calculate(self, input_image):
      pass
      
      
      
      class Process(Process):

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