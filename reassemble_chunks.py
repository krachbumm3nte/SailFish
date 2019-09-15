import io_utils
import logger
import settings
import os
import numpy as np
import SimpleITK as sitk
import sys
import utils
import json

if __name__ == "__main__":
    args = sys.argv[1:]


    with open(args[0]) as json_file:
        configuration = json.load(json_file)
        settings.init(configuration)

    out_name = configuration['file_name']
    temp_name = "{}_*.hdf5".format(configuration['temp_dir'] + out_name)

    image = io_utils.reassemble_chunks(temp_name, out_name, configuration)

    # io_utils.show_3D_image(sitk.GetImageFromArray(result))
