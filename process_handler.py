import io_utils
import logger
import processes
import h5py


class ProcessHandler:

    def __init__(self, process_dicts, rescaling_factor):
        self.processes = [self.create_process_from_dict(p) for p in process_dicts]
        self.rescaling_factor = rescaling_factor
        self.current_shape_unscaled = (0, 0, 0)
        self.current_indices = (0, None)


    def execute_process_list(self, image, chunking=False):
        for i in range(len(self.processes)):
            p = self.processes[i]
            image = p.execute(image, chunking)

        return image


    def create_process_from_dict(self, process_dict):
        if 'type' not in process_dict.keys():
            logger.log_error("process type not specified! validate process definition.")
        process_class = getattr(processes, process_dict['type'])
        instance = process_class(process_dict)
        instance.set_handler(self)
        return instance

    def set_indices(self, start, end, input_location):
        # TODO: This method is fucked
        self.current_indices = (start, end)
        with h5py.File(input_location, 'r') as infile:
            data = None
            for identifier in io_utils.valid_dataset_identifiers:
                if identifier in infile.keys():
                    data = infile[identifier]
                    self.current_shape_unscaled = ((data.shape[0] if end is None else end)-start, data.shape[1], data.shape[2])
                    print(f'new scale: {self.current_shape_unscaled}')
                    break
            if not data:
                logger.log_error(f'unable to extract data from input file: {input_location}')

