import ct_scan
import sys

import logger
import utils
import SimpleITK as sitk
import tkinter_utils
import json
import settings



if __name__ == "__main__":
    args = sys.argv[1:]
    tkinter_utils.tk_init("foo")


    with open(args[0]) as json_file:
        configuration = json.load(json_file)
        settings.init(configuration)


    process_list = utils.create_process_list_from_dict(configuration['processes'])
    logger.log_successful("input file validation")

    tkinter_utils.tk_init("foo")
    if not settings.is_chunking:
        logger.log_timestamp(
            'loading dataset from {} to {}, scaling down by a factor of {}...'.format(settings.start, settings.end, settings.rescaling_factor))

        scan = ct_scan.CtScan(settings.start, settings.end, rescaling_factor=settings.rescaling_factor)
        length, height, width = scan.data.shape
        image = sitk.GetImageFromArray(scan.data)
        logger.log_completed("Image loading")
        image = utils.execute_process_list(process_list, image)


    else:
        chunks = utils.define_chunks(settings.start, settings.end, settings.chunksize)
        logger.log_timestamp(f"chunk generation complete, {len(chunks)} chunks defined.")

        for i in range(len(chunks)):
            chunk = chunks[i]
            logger.log_started(f"loading chunk from index {chunk[0]} to {chunk[1]}")
            scan = ct_scan.CtScan(chunk[0], chunk[1], settings.rescaling_factor)
            image = sitk.GetImageFromArray(scan.data)
            image = utils.execute_process_list(process_list, image, True)
            logger.log_completed(f"processing chunk from index {chunk[0]} to {chunk[1]}")
            utils.write_to_temp(sitk.GetImageFromArray(sitk.GetArrayFromImage(image)[0:-1]), f"chunk_{str(i).zfill(3)}")

    utils.ask_save_image(image)
    utils.show_3D_image(image)

