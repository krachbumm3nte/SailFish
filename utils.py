import numpy as np
import matplotlib.pyplot as plt
from ct_scan import CtScan
import settings


def show_histogram(ct_scan: CtScan):
    """display a histogram of the data encapsulated inside a CtScan object
    :param ct_scan: the input scan
    """
    f = plt.hist(ct_scan.data.ravel(), bins=255, range=(1, 255))
    f.figsize = (12, 9)
    plt.show()


def show_comparative_plot(ct_scan, startindex=0, interval=20):
    """displays a sample of 3 * 3 images from a given CtScan object, starting at startindex and spacing samples by inteval
    :param ct_scan: the input scan to sample from
    :param startindex: index of the first sample
    :param interval: spacing between taking samples
    """
    wh = 3
    num_plots = wh ** 2
    func = np.vectorize(lambda e: e)  # 0 if e<60 else e)
    f, axes = plt.subplots(wh, wh, sharex='col', sharey='row')
    f.dpi = 300
    f.figsize = 15, 15

    for i in range(startindex, startindex + num_plots * interval, interval):
        xpos, ypos = int(i / wh), i % wh
        axes[xpos, ypos].imshow(func(ct_scan.data[i]))
        axes[xpos, ypos].title = f"index {i}"
    plt.show()


def print_image_properties(image):
    print(f"image min:{np.min(image)}, max:{np.max(image)}, mean:{np.mean(image)}, median:{np.median(image)}")


def parse_seeds(seeds, scale):
    return [(int(x/scale), int(y/scale), int(z/scale)) for (x, y, z) in seeds]


def define_chunks(start, end, chunksize):
    # TODO: proper handling of rescaling

    result = []
    for i in range(start, end, chunksize):
        last_index = i + chunksize
        if last_index > end:
            result.append((i, end))
        else:
            result.append((i, last_index))
    return result


def askyesno(question, default):
    y = 'y' if not default else 'Y'
    n = 'n' if default else 'N'
    query = '%s [%s/%s]' % (question, y, n)
    if settings.noconfirm:
        print(query)
        print("override through \"noconfirm\" argument")
        return default
    while True:
        response = input(query)
        if not response:
            return default
        r_char = response[0].lower()
        if r_char == 'y':
            return True
        elif r_char == 'n':
            return False
        else:
            print('are you even trying?')



