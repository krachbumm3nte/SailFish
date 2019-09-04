import logger
import settings
import os
import numpy as np
import SimpleITK as sitk
import sys
import utils
import json
import tkinter_utils

if __name__ == "__main__":
    args = sys.argv[1:]
    tkinter_utils.tk_init('foo')


    with open(args[0]) as json_file:
        configuration = json.load(json_file)
        settings.init(configuration)

    result = None
    print(settings.tempdir)
    print(sorted(os.listdir(settings.tempdir)))
    for file in sorted(os.listdir(settings.tempdir)):
        logger.log_timestamp(f"processing: {file}")
        image = sitk.ReadImage(settings.tempdir + file)
        if result is None:
            result = sitk.GetArrayFromImage(image)
        else:
            result = np.concatenate((result, sitk.GetArrayFromImage(image)), axis=0)

    utils.ask_save_image(sitk.GetImageFromArray(result))
    utils.show_3D_image(sitk.GetImageFromArray(result))
