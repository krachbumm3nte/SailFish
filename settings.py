from datetime import datetime
import os
import sys
import time


def init(configuration_dict):
    global configuration
    configuration = configuration_dict

    global imgdir
    imgdir = retrieve_attribute('image_dir', os.path.curdir)

    global tempdir
    tempdir = retrieve_attribute('temp_dir', imgdir + 'temp/')

    global FILE_ENDING
    FILE_ENDING = retrieve_attribute('file_ending', '.hdf5')

    global rescaling_factor
    rescaling_factor = retrieve_attribute('rescaling_factor', 1)

    global start
    start = retrieve_attribute('start_index')

    global end
    end = retrieve_attribute('end_index')

    global is_chunking
    is_chunking = retrieve_attribute('chunking', False)

    global chunksize
    chunksize = retrieve_attribute('chunksize', 500)

    global start_time
    start_time = time.time()

    global file_identifier
    file_identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def retrieve_attribute(key, default=None):
    if key in configuration.keys():
        return configuration[key]
    elif default is not None:
        print(f"{key} was not defined in the input file, applying default value: {default}")
        return default
    else:
        print(f'Error during input file reading: required attribute not found ({key})')
        sys.exit()