import h5py


class CtScan:
    # TODO: update to match current architecture
    default_path = '/home/johannes/Desktop/billfish/SFA3.original.4v0.hdf5'

    def __init__(self, start=0, end=None, rescaling_factor=1, path=default_path):
        with h5py.File(path, 'r') as f:
            data_orig = f['original']
            meta_grp = f['meta']
            self.height = meta_grp.attrs['height']
            self.width = meta_grp.attrs['width']
            self.length = meta_grp.attrs['length']
            self.data_type = meta_grp.attrs['data type']
            self.resolution = meta_grp.attrs['resolution']
            if end is None:
                end = self.length
            self.data = data_orig[start:end:rescaling_factor, ::rescaling_factor, ::rescaling_factor]
