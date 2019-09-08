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

    result = None
    print(settings.tempdir)
    print(sorted(os.listdir(settings.tempdir)))
    for file in sorted(os.listdir(settings.tempdir)):
        logger.log_timestamp(f"processing: {file}")
        image = sitk.ReadImage(settings.tempdir + file)
        print(image.GetSize())
        if result is None:
            result = sitk.GetArrayFromImage(image)
        else:
            result = np.concatenate((result, sitk.GetArrayFromImage(image)), axis=0)

    io_utils.write_to_file(sitk.GetImageFromArray(result))
    io_utils.show_3D_image(sitk.GetImageFromArray(result))
