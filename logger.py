import time
import sys


class Logger:

    def __init__(self):
        self.start_time = time.time()

    def get_progressed_time(self):
        """
        returns the time progressed since set_start_time() was called as a formatted string for logging use
        :return: a formatted string intended to proceed all logging output
        """
        progressed = time.time()-self.start_time
        return '%.1fs:\t' % progressed

    def log_started(self, process):
        self.log_timestamp(f"{process} started...")

    def log_completed(self, process):
        self.log_timestamp(f"{process} completed.\n")

    def log_successful(self, process):
        self.log_timestamp(f"{process} successful!\n")

    def log_timestamp(self, log_string):
        print(f"{self.get_progressed_time()}{log_string}")

    def log_warning(self, warning):
        print(f"\t\tWaring: {warning}")

    def log_error(self, errormsg):
        print(f"\t\tError: {errormsg}\nExiting.")
        sys.exit()

