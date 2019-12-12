import argparse
import glob

import io_utils
import json
import logger

if __name__ == "__main__":
    io_handler = io_utils.IOHandler(False, logger.Logger())

    parser = argparse.ArgumentParser(description='reassembles the temporary files of a computation into a '
                                                 'complete image and saves it to disk')
    parser.add_argument('configuration', help='the path of the .json file that was used to create the images. Required ')

    parser.add_argument('-nd', '--nodisplay', action='store_true',
                        help='reassembled image will not be displayed')

    parser.add_argument('-ns', '--nosave', action='store_true',
                        help='reassembled image will not be saved to disk')

    args = parser.parse_args()

    with open(args.configuration) as json_file:
        configuration = json.load(json_file)

    out_name = configuration['file_name']
    temp_name = "{}{}*.hdf5".format(configuration['temp_dir'], out_name)
    out_dir = io_handler.define_out_name(configuration['image_dir'], out_name)
    images = list(sorted(glob.iglob(temp_name)))
    image = io_handler.reassemble_chunks(images)

    if not args.nosave:
        io_handler.write_array_to_file(image, out_dir, configuration)

    if not args.nodisplay:
        io_handler.show_3D_array(image)
