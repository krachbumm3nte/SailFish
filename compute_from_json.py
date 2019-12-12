import os
import sys
import json
import argparse

import io_utils
from logger import Logger
from process_handler import ProcessHandler


class JsonInterpreter:
    """
    Takes the location of a .json file as an input, and if the file is valid by the standard specified in the paper,
    computes the specified image operations and saves the output to a new file

    Attributes:
        logger (Logger): encapsulates the output of basic user information during computation
        io_handler (IOHandler): used for reading and writing files
        configuration (dict): contains the entire input file as reference for adding metadata to the output
        input_dir (str): the location of the input file that the computation is supposed to be applied to
        file_identifier (str, optional): an identifier that will be used for naming output and temporary files.
            Defaults to 'visualization'
        output_name (str): the full file location and name for the final output
        temp_name (str): the full file location and name for temporary files. contains a space for formatting an
            index before the file extension
        rescaling_factor (int, optional): the amount of downscaling to be applied to the original image (taking every
            n-th value along all axes). Defaults to 1.
        input_shape_unscaled (int): the shape of the entire input file before rescaling
        start_index, end_index (int, optional): the range of the input file (before scaling) to be computed.
            Defaults to None (loading the entire dataset).
        handler (ProcessHandler): creates and manages the pipeline of processes to be applied to the input image
        chunking (bool, optional): if True, the the Dataset is read in chunks. Every chunk will be transformed
            according to the instructions and saved to disk for later reassembly. Defaults to True.
        chunks (list(tuple), optional): the indices describing all chunks (befor scaling) to be computed
    """
    def __init__(self):
        parser = argparse.ArgumentParser(description='performs computations defined in an input file.')
        parser.add_argument('configuration', help='a valid .json file that contains computation instructions')

        parser.add_argument('-nc', '--noconfirm', action='store_true',
                            help='skips all user input queries and performs their default action')

        parser.add_argument('-nd', '--nodisplay', action='store_true',
                            help='Calculation output will not be displayed')

        parser.add_argument('-ns', '--nosave', action='store_true',
                            help='Calculation output will not be saved to disk')

        self.args = parser.parse_args()

        self.logger = Logger()
        self.io_handler = io_utils.IOHandler(self.args.noconfirm, self.logger)

        with open(self.args.configuration) as json_file:
            self.configuration = json.load(json_file)

        self.input_dir = self.get_attr_by_name('input_dir')
        self.file_identifier = self.get_attr_by_name('file_name', 'result')
        self.output_name = self.io_handler.define_out_name(self.get_attr_by_name('image_dir'), self.file_identifier)
        self.temp_name = "{}_{{}}.hdf5".format(self.get_attr_by_name('temp_dir') + self.file_identifier)
        self.rescaling_factor = self.get_attr_by_name('rescaling_factor', 1)
        self.input_shape_unscaled = self.io_handler.get_original_shape(self.input_dir)
        self.start_index = self.get_attr_by_name('start_index', 0)
        self.end_index = self.get_attr_by_name('end_index', self.input_shape_unscaled[0])

        self.handler = ProcessHandler(self, self.get_attr_by_name('processes'))
        self.chunking = self.get_attr_by_name('chunking', False)
        if self.chunking:
            self.chunks = define_chunks(self.start_index, self.end_index, self.get_attr_by_name('chunksize', 500))
            self.logger.log_timestamp(f"chunk generation complete, {len(self.chunks)} chunks defined.")

        self.logger.log_successful("input file validation")

    def execute(self):

        if len(self.handler.processes) == 0:
            self.logger.log_timestamp('No image Operations defined. Displaying unedited input image.')
            self.io_handler.show_3D_array(self.io_handler.load_array_from_file(self.input_dir, self.rescaling_factor,
                                                                               self.start_index, self.end_index))
            return

        if not self.chunking or len(self.chunks) == 1:
            self.handler.set_indices(self.start_index, self.end_index, self.input_dir)
            working_array = self.io_handler.load_array_from_file(self.input_dir, self.rescaling_factor,
                                                                 self.start_index, self.end_index)
            working_array = self.handler.execute_process_list(working_array)
            self.io_handler.write_array_to_file(working_array, self.output_name, self.configuration)

        else:
            temp_image_dirs = []

            for i in range(len(self.chunks)):
                chunk = self.chunks[i]
                self.handler.current_chunk = i
                self.handler.set_indices(chunk[0], chunk[1], self.input_dir)

                self.logger.log_started(f"loading chunk No. {i} from index {chunk[0]} to {chunk[1]}")
                working_array = self.io_handler.load_array_from_file(self.input_dir, self.rescaling_factor, chunk[0], chunk[1])
                working_array = self.handler.execute_process_list(working_array, True)
                self.logger.log_completed(f"processing chunk from index {chunk[0]} to {chunk[1]}")
                current_temp_name = self.temp_name.format(str(i).zfill(3))
                self.io_handler.write_array_to_file(working_array, current_temp_name, self.configuration)
                temp_image_dirs.append(current_temp_name)

            working_array = self.io_handler.reassemble_chunks(temp_image_dirs)
            self.io_handler.write_array_to_file(working_array, self.output_name, self.configuration)

            if self.io_handler.askyesno("delete temporary files?", True):
                for file in temp_image_dirs:
                    self.logger.log_timestamp(f"deleting: {file}")
                    os.unlink(file)

        if not self.args.nodisplay:
            if self.io_handler.askyesno("\ndisplay output?", True):
                self.io_handler.show_3D_array(working_array)

    def get_attr_by_name(self, key, default=None):
        if key in self.configuration.keys():
            return self.configuration[key]

        if default is not None:
            self.logger.log_timestamp(f"{key}\twas not defined in the input file, applying default value: {default}")
            return default

        self.logger.log_error(f'Error during input file reading: required attribute not found ({key})')
        sys.exit()


def define_chunks(start, end, chunksize):
    result = []
    for i in range(start, end, chunksize):
        last_index = i + chunksize
        if last_index > end:
            result.append((i, end))
        else:
            result.append((i, last_index))
    return result


if __name__ == '__main__':
    foo = JsonInterpreter()
    foo.execute()

