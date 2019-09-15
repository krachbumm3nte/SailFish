import glob
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import utils
import SimpleITK as sitk
import h5py
import numpy as np
import vtk
import os
import re
from PIL import ImageFont, Image, ImageDraw

import settings
from ct_scan import CtScan
import logger

valid_dataset_identifiers = ['data', 'original']


def write_array_to_file(out_array, output_directory, meta):
    """
    writes an itk image object to a given directory
    """
    logger.log_timestamp(f"writing image of size {out_array.shape} to file: {output_directory}...")

    with h5py.File(output_directory, 'w') as outfile:
        outfile.create_dataset('data', shape=out_array.shape, data=out_array)
        meta_grp = outfile.create_group('meta')
        for k, v in meta.items():
            if k == 'processes':
                v = str(v)
            meta_grp.attrs[k] = v
    logger.log_completed("image saving")


def write_to_file(image, output_directory, meta):
    write_array_to_file(sitk.GetArrayFromImage(image), output_directory, meta)


def show_3D_image(image):
    """
    displays an itk-Image class as a 3D-Volume that can be interactivle inspected
    :param image: the input image to be displayed
    """
    imgArray = sitk.GetArrayFromImage(image)
    data_type = image.GetPixelIDTypeAsString()
    dims = imgArray.shape
    data_min, data_max = (np.min(imgArray), np.max(imgArray))

    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    interactor.SetRenderWindow(window)
    renderer.SetBackground(0, 0, 0)
    window.SetSize(800, 600)

    importer = vtk.vtkImageImport()
    if data_type == '8-bit unsigned integer':
        importer.SetDataScalarTypeToUnsignedChar()
    elif data_type == '16-bit unsigned integer':
        importer.SetDataScalarTypeToShort()
    else:
        print(f"unsupported data type ({data_type}) for image display, exiting!")
        exit(0)

    importer.SetNumberOfScalarComponents(1)
    importer.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    importer.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    importer.CopyImportVoidPointer(imgArray, imgArray.nbytes)

    acf = vtk.vtkPiecewiseFunction()
    acf.AddPoint(data_min, 0.0)
    acf.AddPoint(data_max, 1.0)

    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(data_max, 1.0, 1.0, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(colorFunc)
    volume_property.SetScalarOpacity(acf)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputConnection(importer.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    renderer.AddVolume(volume)
    renderer.SetBackground(25.0 / 255.0, 51.0 / 255.0, 102.0 / 255.0)

    window.Render()

    interactor.Initialize()
    interactor.Start()


def load_array_from_file(filename, rescaling_factor=1, start_index=0, end_index=None):
    logger.log_info(f'loading image data at index {start_index} to {end_index} from file: {filename}')
    with h5py.File(filename, 'r') as infile:
        data = None
        for identifier in valid_dataset_identifiers:
            if identifier in infile.keys():
                data = infile[identifier]
                break
        if not data:
            logger.log_error(f'unable to extract data from input file: {filename}')

        print(data.dtype)
        original_shape = ((end_index if end_index is not None else data.shape[0]) - start_index, data.shape[1], data.shape[2])
        print(original_shape)
        if rescaling_factor != 1:
            image = data[start_index:end_index:rescaling_factor, ::rescaling_factor, ::rescaling_factor]
            print("rescaling input")
            print(image.shape)
        else:
            image = data[start_index:end_index]
        logger.log_info(f"finished loading from file. resulting image size: {image.shape}")
    return image


def load_from_file(filename, scale, start_index=0, end_index=None):
    return sitk.GetImageFromArray(load_array_from_file(filename, scale, start_index, end_index))


def reassemble_chunks(temp_regex, out_name, meta):
    logger.log_info(f"beginning chunk reassembly from {temp_regex}")

    images = sorted(glob.iglob(temp_regex))
    if len(images) == 0:
        logger.log_error("no temporary files detected")
    result = assemble_images(images)
    write_to_file(sitk.GetImageFromArray(result), out_name, meta)

    if utils.askyesno("delete temporary files?", True):
        for file in images:
            logger.log_info(f"deleting: {file}")
            os.unlink(file)

    return sitk.GetImageFromArray(result)


def assemble_images(image_locations):
    result = None
    for location in image_locations:
        logger.log_timestamp(f"processing: {location}")
        current_image_data = load_array_from_file(location)
        print(current_image_data.shape)
        if result is None:
            result = current_image_data
        else:
            result = np.concatenate((result, current_image_data), axis=0)

    logger.log_info(f"Assembly finished, resulting size: {result.shape}")
    return result


def define_out_name(directory, name):

    output_directory = "{}{}.hdf5".format(directory, name)
    if os.path.isfile(output_directory):
        logger.log_info('output file already exists, adding unique identifier...')
        unique = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        output_directory = "{}{}{}.hdf5".format(directory, name, unique)
    return output_directory
