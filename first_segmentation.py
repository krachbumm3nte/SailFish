import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import subprocess
from pathlib import Path
import SimpleITK as sitk
import time

imgdir = '/home/johannes/Desktop/billfish/slices/'
tempdir = imgdir + 'temp/'


def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()

def prepare_sitk(img, title=None, margin=0.05, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    return fig


if __name__ == "__main__":
    image_name = imgdir + "2925.png"
    reader = sitk.ImageFileReader()
    reader.SetImageIO("PNGImageIO")
    reader.SetFileName(image_name)
    image = reader.Execute()

    f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True)
    #plt.set_cmap("gray")

    start = time.time()
    imgSmooth = sitk.CurvatureFlow(image1=image,
                                   timeStep=0.5,
                                   numberOfIterations=5)
    ax0.imshow(sitk.GetArrayFromImage(imgSmooth))

    seeds = [(500, 500)]  # [(470,420), (950,420)]
    internal_structure = sitk.ConnectedThreshold(image1=imgSmooth,
                                                 seedList=seeds,
                                                 lower=35,
                                                 upper=255,
                                                 replaceValue=1)
    ax1.imshow(sitk.GetArrayFromImage(internal_structure))

    imgSmoothInt = sitk.Cast(sitk.RescaleIntensity(imgSmooth), internal_structure.GetPixelID())
    internal_structure_no_holes = sitk.VotingBinaryIterativeHoleFilling(image1=internal_structure,
                                                                        maximumNumberOfIterations=10,
                                                                        radius=[15] * 6,
                                                                        majorityThreshold=1,
                                                                        backgroundValue=0,
                                                                        foregroundValue=1)
    ax2.imshow(sitk.GetArrayFromImage(internal_structure_no_holes))
    imgSmoothMasked = sitk.Mask(imgSmooth, internal_structure_no_holes)

    ax3.imshow(sitk.GetArrayFromImage(imgSmoothMasked))
    #ax3.imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(imgSmoothInt, internal_structure_no_holes)))

    end = time.time()
    print(end - start)
    plt.show()
