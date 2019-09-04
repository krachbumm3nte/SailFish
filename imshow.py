import utils
import SimpleITK as sitk

image = sitk.ReadImage('/home/johannes/Desktop/billfish/plots/visualization_2019_08_28_16_13_34.hdf5')
utils.show_3D_image(image)