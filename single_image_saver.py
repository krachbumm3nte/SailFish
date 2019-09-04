if __name__ == '__main__':

    import os
    import h5py
    import numpy as np
    import png

    dataset = 'SFA3'
    path = '/home/johannes/PycharmProjects/SailFish/res'
    name = '{0}.original.4v0.hdf5'.format(dataset)
    data_start = 1100
    data_length = 30

    archive = os.path.join(path, name)
    print('archive = {0}'.format(archive))
    with h5py.File(archive, 'r') as f:
        original_dset = f['original']
        meta_grp = f['meta']
        height = meta_grp.attrs['height']
        width = meta_grp.attrs['width']
        length = meta_grp.attrs['length']
        data_type = meta_grp.attrs['data type']
        resolution = meta_grp.attrs['resolution']

        print('W x H x L = {0} x {1} x {2}'.format(width, height, length))
        print('resolution = {0} µm/Voxel'.format(resolution))
        print('data type = {0}'.format(data_type))
        for i in range(length):
            if i % 50 == 0:
                print(f'{i} images saved')
            png.from_array(np.array(original_dset[i]), mode='L').save(f'/home/johannes/PycharmProjects/SailFish/res/slices/{i}.png')
    print("done.")
