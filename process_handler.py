import logger
import processes
import utils


class ProcessHandler:

    def __init__(self, process_dicts):
        self.processes = self.create_process_list_from_json(process_dicts)
        self.inputs = [p.input for p in self.processes if p is not -1]
        print(self.inputs)


    def execute_process_list(self, image, chunking=False):
        for i in range(len(self.processes)):
            p = self.processes[i]
            if p.input == -1:
                image = p.execute(image, chunking)
            else:
                input_file = utils.load_from_file(p.input)
                image = p.execute(input_file, chunking)
        return image

    def create_process_list_from_json(self, p_list):
        return [self.create_process_from_dict(p) for p in p_list]


    def create_process_from_dict(self, process_dict):
        if 'type' not in process_dict.keys():
            logger.log_error("process type not specified! validate process definition.")
        process_class = getattr(processes, process_dict['type'])
        return process_class(process_dict)