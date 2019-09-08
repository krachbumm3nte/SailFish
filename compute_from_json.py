import ct_scan
import sys

import io_utils
import logger
import utils
import SimpleITK as sitk
import json
import settings
from process_handler import ProcessHandler

if __name__ == "__main__":
    args = sys.argv[1:]


    with open(args[0]) as json_file:
        configuration = json.load(json_file)
        settings.init(configuration)


    if 'noconfirm' in args:
        settings.noconfirm = True
        logger.log_info('noconfirm argument read, skipping all user input queries')

    input_dir = ct_scan.CtScan.default_path
    logger.log_info(f'no input path specified, using default path at: {input_dir}')


    handler = ProcessHandler(configuration['processes'])
    logger.log_successful("input file validation")

    if not settings.is_chunking:
        logger.log_timestamp(
            'loading dataset from {} to {}, scaling down by a factor of {}...'.format(settings.start, settings.end, settings.rescaling_factor))

        scan = ct_scan.CtScan(settings.start, settings.end, rescaling_factor=settings.rescaling_factor, path=input_dir)
        image = sitk.GetImageFromArray(scan.data)
        logger.log_completed("Image loading")
        image = handler.execute_process_list(image)
        io_utils.write_to_file(image)

    else:
        chunks = utils.define_chunks(settings.start, settings.end, settings.chunksize)
        logger.log_timestamp(f"chunk generation complete, {len(chunks)} chunks defined.")

        for i in range(len(chunks)):
            chunk = chunks[i]
            settings.current_chunk = (i, chunk[0], chunk[1])
            logger.log_started(f"loading chunk from index {chunk[0]} to {chunk[1]}")
            image = io_utils.load_from_file(input_dir, chunk[0], chunk[1])
            logger.log_info(f"processing image of size {image.GetSize()}")
            image = handler.execute_process_list(image, True)
            logger.log_completed(f"processing chunk from index {chunk[0]} to {chunk[1]}")
            if i != len(chunks) - 1:
                print('shortening by 1')
                image = image[:, :, 0:-1]
            io_utils.write_to_file(image, f"{settings.file_name}_{str(i).zfill(3)}", True)

        image = io_utils.reassemble_chunks()

    io_utils.show_3D_image(image)

