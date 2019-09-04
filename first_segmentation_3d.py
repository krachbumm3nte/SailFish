import ct_scan
import sys

from utils import *
import SimpleITK as sitk
from tkinter import messagebox
import tkinter_utils
import itk
import json





if __name__ == "__main__":
    args = sys.argv[1:]
    print(sys.argv)
    imgdir = '/home/johannes/Desktop/billfish/slices/'
    tempdir = imgdir + 'temp/'
    rescaling_factor = 3
    start = 0
    end = 1000
    # channel_seeds = [(int(500/RESCALING_FACTOR),int(500/RESCALING_FACTOR),0), (int(1024/RESCALING_FACTOR), int(480/RESCALING_FACTOR), 0)]
    growth_seeds = [(int(456 / rescaling_factor), int(424 / rescaling_factor), 0),
                    (int(930 / rescaling_factor), int(426 / rescaling_factor), 0)]

    if len(args) == 1:
        with open(args[0]) as json_file:
            data = json.load(json_file)
            imgdir = data['image_dir']
            tempdir = data['temp_dir']
            rescaling_factor = data['rescaling_factor']
            start = data['start_index']
            end = data['end_index']
            growth_seeds = parse_seeds(data['growth_seeds'], rescaling_factor)

    tkinter_utils.tk_init("foo")

    scan = ct_scan.CtScan(start, end, rescaling_factor=rescaling_factor)
    log_completed("CT scan reading")
    length, height, width = scan.data.shape
    image = sitk.GetImageFromArray(scan.data)
    log_completed("Image object generation")

    data_type = scan.data_type

    pixelID = image.GetPixelID()

    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(0.3)
    imgTemp = gaussian.Execute(image)

    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    imgSmooth = caster.Execute(imgTemp)

    log_completed("imagesmoothing")

    seeds = [(int(width/2), int(height/2), 0)]
    internal_structure = sitk.ConnectedThreshold(image1=imgSmooth,
                                                 seedList=seeds,
                                                 lower=35,
                                                 upper=255,
                                                 replaceValue=1)
    log_completed("structure generation via connected threshold")
    print_image_properties(internal_structure)

    imgSmoothInt = sitk.Cast(sitk.RescaleIntensity(imgSmooth), internal_structure.GetPixelID())
    log_completed("rescaling")
    """
    internal_structure_no_holes = sitk.VotingBinaryIterativeHoleFilling(image1=internal_structure,
                                                                        maximumNumberOfIterations=5,
                                                                        radius=[5] * 4,
                                                                        majorityThreshold=1,
                                                                        backgroundValue=0,
                                                                        foregroundValue=1)
    logProgress("hole filling")
    print_image_properties(internal_structure_no_holes)
    
    imgSmoothMasked = sitk.Mask(image, internal_structure_no_holes)
    logProgress("masking")
    """
    channels_grown = sitk.ConnectedThreshold(image1=image,
                                             seedList=growth_seeds,
                                             lower=5,
                                             upper=42)
    log_completed("region growing")

    show_3D_image(sitk.GetArrayFromImage(imgSmoothInt), data_type)

    if messagebox.askyesno(title="Save to file?", message="Save this visualization on your disk?"):

        filename = "{}internal_structure_{}-{}_scale-{}.hdf5".format(imgdir, start, end, rescaling_factor)
        write_to_file(channels_grown, filename)

