import argparse

import io_utils
from logger import Logger

if __name__ == "__main__":
    handler = io_utils.IOHandler(False, Logger())

    parser = argparse.ArgumentParser(description='displays a .hdf5 file interactively.')
    parser.add_argument('infile',
                        help=f'a valid .hdf5 file that contains a dataset by the key of either: {handler.valid_dataset_identifiers}')

    parser.add_argument('--si', '--start_index', type=int, action='store', default=0,
                        help='the start index along the X-Axis of the '
                             'volume to be displayed')
    parser.add_argument('--ei', '--end_index', type=int, default=None, help='the end index along the X-Axis of the '
                                                                            'volume to be displayed')

    parser.add_argument('--s', '--scale', type=int, default=1, help='The downscaling to be applied to the image before'
                                                                    'it is being displayed')

    args = parser.parse_args()

    image = handler.load_array_from_file(args.infile, args.s, args.si, args.ei)

    handler.show_3D_array(image)
