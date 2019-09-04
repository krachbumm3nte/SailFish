# -*- coding: utf-8 -*-
'''
Created on 07.06.2019

@author: breukerm
'''


if __name__ == '__main__':
    import time
    import h5py
    import vtk
    import numpy as np
    from skimage.filters import threshold_otsu

    start= time.time()
    print('started...')

    dataset = 'SFA3'
    path = 'D:/billfish/4v0/{0}'.format(dataset)
    name = '{0}.original.4v0.hdf5'.format(dataset)
    data_start = 0
    data_length = 500

    archive = 'res/SFA3.original.4v0.hdf5'
    print('archive = {0}'.format(archive))
    with h5py.File(archive, 'r') as f:
        original_dset = f['original']
        meta_grp = f['meta']
        height = meta_grp.attrs['height']
        width = meta_grp.attrs['width']
        length = meta_grp.attrs['length']
        data_type = meta_grp.attrs['data type']
        resolution = meta_grp.attrs['resolution']

        print('L x H x W = {0} x {1} x {2}'.format(length, height, width))
        print('resolution = {0} um/Voxel'.format(resolution))
        print('data type = {0}'.format(data_type))

        data = original_dset[::5, ::5, ::5]
        # data = original_dset[data_start: data_start + data_length]

    dims = data.shape
    print(data.shape)
    data_min, data_max = np.min(data), np.max(data)
    data_otsu = threshold_otsu(data.ravel())

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
    if data_type == 'uint8':
        importer.SetDataScalarTypeToUnsignedChar()
    elif data_type == 'uint16':
        importer.SetDataScalarTypeToShort()
    else:
        exit(0)
    importer.SetNumberOfScalarComponents(1)
    importer.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    importer.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    importer.CopyImportVoidPointer(data, data.nbytes)

    acf = vtk.vtkPiecewiseFunction()
    acf.AddPoint(data_min, 0.0)
    acf.AddPoint(data_otsu, 0.0)
    acf.AddPoint(data_max, 1.0)

    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(data_max, 1.0, 1.0, 1.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(acf)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputConnection(importer.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer.AddVolume(volume)
    renderer.SetBackground(25.0 / 255.0, 51.0 / 255.0, 102.0 / 255.0)

    window.Render()

    print('finished. {0:1.1f}s'.format(time.time() - start))

    interactor.Initialize()
    interactor.Start()
