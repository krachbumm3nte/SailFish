import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import subprocess
from pathlib import Path
from ct_scan import CtScan
import vtk
import time
from datetime import datetime
import SimpleITK as sitk
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import settings
import processes
import h5py
import logger

from logger import get_progressed_time, log_completed


def show_histogram(ct_scan: CtScan):
    """display a histogram of the data encapsulated inside a CtScan object
    :param ct_scan: the input scan
    """
    f = plt.hist(ct_scan.data.ravel(), bins=255, range=(1, 255))
    f.figsize = (12, 9)
    plt.show()


def show_comparative_plot(ct_scan, startindex=0, interval=20):
    """displays a sample of 3 * 3 images from a given CtScan object, starting at startindex and spacing samples by inteval
    :param ct_scan: the input scan to sample from
    :param startindex: index of the first sample
    :param interval: spacing between taking samples
    """
    wh = 3
    num_plots = wh ** 2
    func = np.vectorize(lambda e: e)  # 0 if e<60 else e)
    f, axes = plt.subplots(wh, wh, sharex='col', sharey='row')
    f.dpi = 300
    f.figsize = 15, 15

    for i in range(startindex, startindex + num_plots * interval, interval):
        xpos, ypos = int(i / wh), i % wh
        axes[xpos, ypos].imshow(func(ct_scan.data[i]))
        axes[xpos, ypos].title = f"index {i}"
    plt.show()


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
    data_min, data_max = np.min(imgArray), np.max(imgArray)

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

    print(data_min, data_max, np.mean(imgArray))
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


def print_image_properties(image):
    print(f"image min:{np.min(image)}, max:{np.max(image)}, mean:{np.mean(image)}, median:{np.median(image)}")


def write_to_file(image: sitk.Image, filedir):
    """
    writes an itk image object to a given directory
    :param image:
    :param filedir:
    """
    filedir = tk.simpledialog.askstring(title="choose filename",
                              prompt="pick the filename and location you want to save this image to:", initialvalue=filedir)
    if filedir:
        print(get_progressed_time() + f"writing image to file: {filedir}...")
        out_array = sitk.GetArrayFromImage(image)
        with h5py.File(filedir, 'w') as outfile:
            outfile.create_dataset('data', shape=out_array.shape, data=out_array)
            meta_grp = outfile.create_group('meta')
            for k, v in settings.configuration.items():
                if k != 'processes':
                    print(k, v)
                    meta_grp.attrs[k] = v

            print(outfile.keys())
            print(meta_grp.keys())
        outfile.close()
        log_completed("image saving")
    else:
        print("image saving cancelled.")

def load_from_file(filename):
    with h5py.File(filename, 'r') as infile:
        image = infile['data'][:]
    return sitk.GetImageFromArray(image)

def ask_save_image(image, filename=None):
    if messagebox.askyesno(title="Save to file?", message="Save this visualization on your disk?"):
        if filename is None:
            filename = "{}visualization_{}.hdf5".format(settings.imgdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        write_to_file(image, filename)


def write_to_temp(image, name):
    sitk.WriteImage(image, settings.tempdir + name + settings.FILE_ENDING)
    log_completed('saving to temporary file')


def parse_seeds(seeds, scale):
    return [(int(x/scale), int(y/scale), int(z/scale)) for (x, y, z) in seeds]





def define_chunks(start, end, chunksize):
    result = []
    for i in range(start, end, chunksize):
        last_index = i + chunksize + 1
        if last_index > end:
            result.append((i, end))
        else:
            result.append((i, last_index))
    return result




