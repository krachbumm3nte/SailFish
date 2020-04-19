import h5py
import SimpleITK as sitk

import processes


class ProcessHandler:
    """
    manages the creation and execution of the list of processes to be applied to the image

    Attributes:
        json_interpreter (JsonInterpreter): The JsonInterpreter that instantiated this object. Used to extract runtime
            variables to avoid duplicate reading/instantiating
        logger (Logger): encapsulates the output of basic user information during computation
        processes (list): a list of objects inheriting from Process that will be executed in order
        rescaling_factor (int): the amount of downscaling to be applied to the original image (taking every
            n-th value along all axes).
        current_chunk (int): the index of the current chunk being processed (if chunking is enabled). Used to determine
            the current state by the processes
        current_shape_unscaled ((int, int, int)): the shape of the current chunk (or whole image) before downscaling
        current_indices (int, int): start and end indexes along the X-Axis of the part of the image currently processed
        io_handler (IOHandler): used for reading and writing files
    """

    def __init__(self, json_interpreter, process_dicts):
        self.json_interpreter = json_interpreter
        self.logger = self.json_interpreter.logger
        self.processes = [self.create_process_from_dict(p) for p in process_dicts]
        self.rescaling_factor = self.json_interpreter.rescaling_factor
        self.current_chunk = 0
        self.current_shape_unscaled = (0, 0, 0)
        self.current_indices = (0, None)
        self.io_handler = json_interpreter.io_handler

    def execute_process_list(self, in_array, chunking=False):
        """
        applies all processes to a given array, and returns a transformed array
        :param in_array: the array to be manipulated
        :param chunking: if True, processes will treat the array as part of a larger image
        :return: a modified array
        """
        image = sitk.GetImageFromArray(in_array)
        for i in range(len(self.processes)):
            p = self.processes[i]
            image = p.execute(image, chunking)
            if p.show_output:
                self.io_handler.show_3D_image(image)

        return sitk.GetArrayFromImage(image)

    def create_process_from_dict(self, process_dict):
        """
        Creates an object of the type Process from a dict of attributes
        :param process_dict: the dict to be encrypted
        :return: an object of type Process
        """
        if 'type' not in process_dict.keys():
            self.logger.log_error("process type not specified! validate process definition.")
        process_class = getattr(processes, process_dict['type'])
        instance = process_class(process_dict)
        instance.set_master(self)
        instance.set_logger(self.logger)
        return instance

    def set_indices(self, start, end, input_location):
        self.current_indices = (start, end)
        with h5py.File(input_location, 'r') as infile:
            data = None
            for identifier in self.io_handler.valid_dataset_identifiers:
                if identifier in infile.keys():
                    data = infile[identifier]
                    self.current_shape_unscaled = (end-start, data.shape[1], data.shape[2])
                    break
            if not data:
                self.logger.log_error(f'unable to extract data from input file: {input_location}')

