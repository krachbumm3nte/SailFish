import ct_scan
import sys

import io_utils
import logger
import utils
import json
import settings
from process_handler import ProcessHandler




def retrieve_attribute(key, default=None):
    if key in configuration.keys():
        return configuration[key]
    elif default is not None:
        print(f"{key} \twas not defined in the input file, applying default value: {default}")
        return default
    else:
        print(f'Error during input file reading: required attribute not found ({key})')
        sys.exit()


if __name__ == "__main__":
    args = sys.argv[1:]


    with open(args[0]) as json_file:
        configuration = json.load(json_file)
        settings.init(configuration)


    if 'noconfirm' in args:
        settings.noconfirm = True
        logger.log_info('noconfirm argument read, skipping all user input queries')

    input_dir = retrieve_attribute('input_dir', ct_scan.CtScan.default_path)


    rescaling_factor = retrieve_attribute('rescaling_factor', 1)

    start_index = configuration['start_index']

    end_index = configuration['end_index']
    file_name = retrieve_attribute('file_name', 'visualization')
    temp_name = "{}_{{}}.hdf5".format(retrieve_attribute('temp_dir') + file_name)
    out_name = io_utils.define_out_name(retrieve_attribute('image_dir'), file_name)

    handler = ProcessHandler(configuration['processes'], rescaling_factor)
    logger.log_successful("input file validation")


    if not retrieve_attribute('chunking', True):
        handler.set_indices(start_index, end_index, input_dir)
        logger.log_timestamp('loading dataset from {} to {}, scaling down by a factor of {}...'
                             .format(start_index, end_index, rescaling_factor))

        image = io_utils.load_from_file(input_dir, rescaling_factor, start_index, end_index)
        logger.log_completed("Image loading")
        image = handler.execute_process_list(image)
        io_utils.write_to_file(image, out_name, configuration)

    else:
        chunks = utils.define_chunks(start_index, end_index, retrieve_attribute('chunksize', 500))
        logger.log_timestamp(f"chunk generation complete, {len(chunks)} chunks defined.")

        for i in range(len(chunks)):
            chunk = chunks[i]
            handler.set_indices(chunk[0], chunk[1], input_dir)
            print(f"current indices: {handler.current_indices}")
            logger.log_started(f"loading chunk from index {chunk[0]} to {chunk[1]}")
            image = io_utils.load_from_file(input_dir, rescaling_factor, chunk[0], chunk[1])
            image = handler.execute_process_list(image, True)
            logger.log_completed(f"processing chunk from index {chunk[0]} to {chunk[1]}")
            io_utils.write_to_file(image, temp_name.format(str(i).zfill(3)), configuration)


        image = io_utils.reassemble_chunks(temp_name.format('*'), out_name, configuration)

    # io_utils.show_3D_image(image)
