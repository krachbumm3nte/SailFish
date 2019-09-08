import sys

from io_utils import show_3D_image
import SimpleITK as sitk
import h5py

if __name__ == "__main__":

    with h5py.File(sys.argv[1], 'r') as infile:
        image = infile['data'][:]
    image = sitk.GetImageFromArray(image)
    show_3D_image(image)

