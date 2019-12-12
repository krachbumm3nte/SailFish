from datetime import datetime
import SimpleITK as sitk
import h5py
import numpy as np
import vtk
import os


class IOHandler:
    """
    A class to manage most IO-operations for the image computation software

    Attributes:
        valid_dataset_identifiers (list): all valid keywords, that indicate a dataset within an .hdf5 file
            This list needs to be expanded to allow the reading of datasets wit different identifiers.
        logger (Logger): encapsulates the output of basic user information during computation
        no_confirm (bool): if true, all user input queries will be skipped, and their default option will be applied
    """

    def __init__(self, no_confirm, logger):
        self.valid_dataset_identifiers = ['data', 'original']
        self.logger = logger
        self.no_confirm = no_confirm
        if self.no_confirm:
            logger.log_timestamp('noconfirm argument read, skipping all user input queries')

    def write_array_to_file(self, out_array, output_directory, meta):
        """
        writes a numpy.ndarray to a given directory
        :param out_array: the array to be written to disk
        :param output_directory: the full directory (including file name and .hdf5)
        :param meta: metadata to be added in the output file
        """
        self.logger.log_timestamp(f"writing image of size {out_array.shape} to file: {output_directory}...")

        with h5py.File(output_directory, 'w') as outfile:
            outfile.create_dataset('data', shape=out_array.shape, data=out_array)
            meta_grp = outfile.create_group('meta')
            self.logger.log_timestamp('adding meta info')
            for k, v in meta.items():
                if k == 'processes':
                    v = str(v)
                meta_grp.attrs[k] = v
        self.logger.log_completed("image saving")

    def write_to_file(self, image, output_directory, meta):
        self.write_array_to_file(sitk.GetArrayFromImage(image), output_directory, meta)

    def show_3D_image(self, image):
        self.show_3D_array(sitk.GetArrayFromImage(image))

    def show_3D_array(self, in_array):
        """
        displays a numpy.ndarray as a 3D-Volume that can be interactively inspected
        :param in_array: the input 3D-array to be displayed
        """
        data_type = in_array.dtype
        dims = in_array.shape
        data_min, data_max = (np.min(in_array), np.max(in_array))
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
            self.logger.log_error(f"unsupported data type ({data_type}) for image display, exiting!")

        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        importer.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        importer.CopyImportVoidPointer(in_array, in_array.nbytes)

        acf = vtk.vtkPiecewiseFunction()
        acf.AddPoint(data_min, 0.0)
        acf.AddPoint(data_max, 1.0)

        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(data_max, 1.0, 1.0, 1.0)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(acf)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputConnection(importer.GetOutputPort())

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        renderer.AddVolume(volume)
        renderer.SetBackground(255, 255, 255)  # (25.0 / 255.0, 51.0 / 255.0, 102.0 / 255.0)

        axes = vtk.vtkAxesActor()

        # swap X and Z-Axes due to VTKs inverted indexing
        axes.SetXAxisLabelText('Z')
        axes.SetZAxisLabelText('X')

        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(interactor)
        # widget.SetViewport(0.0, 0.0, 0.4, 0.4)
        widget.SetEnabled(1)
        widget.InteractiveOn()

        window.Render()

        interactor.Initialize()
        interactor.Start()

    def load_array_from_file(self, filename, rescaling_factor, start_index=0, end_index=None):
        '''
        returns a numpy.ndarray that represents a file given by directory
        :param filename: the directory to be read from
        :param rescaling_factor: the amount of downscaling to be applied to the original image (taking every n-th
            value along all axes
        :param start_index: start index of the image segment to be read
        :param end_index: end index of the image semgent to be read
        :return: numpy array representing the image segment
        '''
        self.logger.log_timestamp(f'loading image data at index {start_index} to {end_index} from file: {filename}')
        with h5py.File(filename, 'r') as infile:
            data = None
            for identifier in self.valid_dataset_identifiers:  # search for valid keys in the file indicating a dataset
                if identifier in infile.keys():
                    data = infile[identifier]
                    break
            if not data:
                self.logger.log_error(f'unable to extract data from input file: {filename}')

            # load the desired section from the dataframe
            if rescaling_factor != 1:
                image = data[start_index:end_index:rescaling_factor, ::rescaling_factor, ::rescaling_factor]
                self.logger.log_timestamp(f"rescaling input by a factor of {rescaling_factor}.")
            else:
                image = data[start_index:end_index]
            self.logger.log_timestamp(f"finished loading from file. resulting image size: {image.shape}")
        return image

    def load_from_file(self, filename, scale, start_index=None, end_index=None):
        return sitk.GetImageFromArray(self.load_array_from_file(filename, scale, start_index, end_index))

    def reassemble_chunks(self, files):
        """
        reassembles a list of files along their x-Axis into a complete image
        :param files: a list of file locations to be concatenated. All files must have an equal shape along ther Y- and Z-Axes
        :return: a complete image containing all files from the input
        """
        self.logger.log_timestamp(f"beginning chunk reassembly from:\n {files}\n")

        if len(files) == 0:
            self.logger.log_error("no temporary files found")

        result = None
        for image_dir in files:
            self.logger.log_timestamp(f"processing: {image_dir}")
            current_image_data = self.load_array_from_file(image_dir, 1)
            if result is None:
                result = current_image_data
            else:
                result = np.concatenate((result, current_image_data), axis=0)

        self.logger.log_timestamp(f"Assembly finished, resulting image size: {result.shape}")
        return result

    def define_out_name(self, directory, name):
        output_directory = "{}{}.hdf5".format(directory, name)
        if os.path.isfile(output_directory):
            self.logger.log_timestamp('output file already exists, adding unique identifier...')
            unique = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            output_directory = "{}{}{}.hdf5".format(directory, name, unique)
        return output_directory

    def askyesno(self, question, default):
        """
        displays a yes/no user promt in console and returns its result
        :param question: the question to be displayed in console
        :param default: boolean - default answer (applied if the user hits enter without answering, or if no_confirm
            is True)
        :return: boolean
        """

        y = 'Y' if default else 'y'
        n = 'n' if default else 'N'  # displaying default answer as a capital letter
        query = '%s [%s/%s]:\n' % (question, y, n)
        if self.no_confirm:  # if the noconfirm argument is read, all calls to this function will return their default value
            print(question)
            print(f"override through \"noconfirm\" argument: {y if default else n}")
            return default

        while True:
            response = input(query)
            if not response:
                return default
            r_char = response[0].lower()
            if r_char == 'y':
                return True
            elif r_char == 'n':
                return False
            else:
                print('try again...')

    def get_original_shape(self, filename):
        """
        returns the unscaled shape of a .hdf5 file, if it contains a dataset with the known valid identifiers
        :param filename: the file to be analyzed
        :return: the shape (x, y, z) of the files main dataset
        """
        with h5py.File(filename, 'r') as infile:
            for identifier in self.valid_dataset_identifiers:
                if identifier in infile.keys():
                    return infile[identifier].shape
        self.logger.log_error('unable to extract data shape from file')

