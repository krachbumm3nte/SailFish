import sys

from io_utils import show_3D_image
import SimpleITK as sitk
import h5py

if __name__ == "__main__":
    scale = 4
    with h5py.File(sys.argv[1], 'r') as infile:
        image = infile['data'][::scale, ::scale, ::scale]
        print(f"loading file of size: {image.shape}")
        for k,v in infile['meta'].attrs.items():
            print(k,v)
    image = sitk.GetImageFromArray(image)
    show_3D_image(image)

