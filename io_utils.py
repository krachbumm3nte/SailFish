import subprocess
from datetime import datetime
from pathlib import Path
import utils
import SimpleITK as sitk
import h5py
import numpy as np
import vtk
import os
from PIL import ImageFont, Image, ImageDraw

import settings
from ct_scan import CtScan
import logger

valid_dataset_identifiers = ['data', 'original']


def write_array_to_file(out_array, name=None, temp=False):
    """
    writes an itk image object to a given directory
    """

    if not name:
        name=settings.file_name

    folder = settings.tempdir if temp else settings.imgdir

    output_directory = "{}{}{}".format(folder, name, settings.FILE_ENDING)
    if os.path.isfile(output_directory):
        logger.log_info('output file already exists, adding unique identifier...')
        unique = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        output_directory = "{}{}{}{}".format(folder, name, unique, settings.FILE_ENDING)

    logger.log_timestamp(f"writing image to file: {output_directory}...")

    with h5py.File(output_directory, 'w') as outfile:
        outfile.create_dataset('data', shape=out_array.shape, data=out_array)
        meta_grp = outfile.create_group('meta')
        for k, v in settings.configuration.items():
            if k == 'processes':
                v = str(v)
            meta_grp.attrs[k] = v
    logger.log_completed("image saving")


def write_to_file(image, name=None, temp=False):
    write_array_to_file(sitk.GetArrayFromImage(image), name, temp)


def save_selection_to_gif(gif_name, start=0, end=10, gif_delay=20, interval=1):
    # TODO: update method head to accept array
    print(f'constructing gif: {gif_name}.gif')
    scan = CtScan(start, end)
    font = ImageFont.truetype(font='usr/share/fonts/TTF/LiberationMono-Bold.ttf', size=35)

    print('generating subimages...')
    for i in range(0, end - start, interval):
        img = Image.fromarray(scan.data[i])
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), str(start + i), 255, font=font)
        draw = ImageDraw.Draw(img)
        img.save(f"{settings.tempdir}{start + i}.png")

    print("collecting images to gif...")
    bash_command = f"convert -delay {gif_delay} {settings.tempdir}*.png {settings.imgdir}{gif_name}.gif"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # delete obsolete images
    print("deleting obsolete images...")
    for p in Path(settings.tempdir).glob("*.png"):
        p.unlink()
    print('done.')


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
        print(f"unsupported data type ({data_type}), exiting!")
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


def load_array_from_file(filename, start_index=0, end_index=-1):
    logger.log_info(f'loading image data at index {start_index} to {end_index} from file: {filename}')
    rescaling_factor = settings.rescaling_factor
    with h5py.File(filename, 'r') as infile:
        data = None
        for identifier in valid_dataset_identifiers:
            if identifier in infile.keys():
                data = infile[identifier]
                break
        if not data:
            logger.log_error(f'unable to extract data from input file: {filename}')
        if 'rescaling_factor' in infile['meta'].attrs.keys():
            rescaling_factor = infile['meta'].attrs['rescaling_factor']
        image = data[start_index:end_index:rescaling_factor, ::rescaling_factor, ::rescaling_factor]
        logger.log_info(f"finished loading from file. resulting image size: {image.shape}")
    return image


def load_from_file(filename, start_index=0, end_index=-1):
    return sitk.GetImageFromArray(load_array_from_file(filename, start_index, end_index))


def ask_save_image(image):
    if utils.askyesno("Save this visualization on your disk?", True):
        write_to_file(image)


def reassemble_chunks():
    logger.log_info(f"beginning chunk reassembly from {settings.tempdir}")
    result = None
    print(sorted(os.listdir(settings.tempdir)))
    for file in sorted(os.listdir(settings.tempdir)):
        logger.log_timestamp(f"processing: {file}")
        current_chunk_data = load_array_from_file(settings.tempdir + file)
        print(current_chunk_data.shape)
        if result is None:
            result = current_chunk_data
        else:
            result = np.concatenate((result, current_chunk_data), axis=0)
    write_to_file(sitk.GetImageFromArray(result))
    return sitk.GetImageFromArray(result)

