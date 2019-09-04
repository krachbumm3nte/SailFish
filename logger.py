import time
import sys
import settings


def get_progressed_time():
    """
    returns the time progressed since set_start_time() was called as a formatted string for logging use
    :return: a formatted string intended to proceed all logging output
    """
    progressed = time.time()-settings.start_time
    return '%.1fs:\t' % progressed


def log_started(process):
    log_timestamp(f"{process} started...")


def log_completed(process):
    log_timestamp(f"{process} completed.")


def log_successful(process):
    log_timestamp(f"{process} successful!")


def log_info(info):
    print(get_progressed_time() + info)


def log_warning(warning):
    print(f"\t\tWaring: {warning}")


def log_error(errormsg):
    print(f"\t\tError: {errormsg}\nExiting.")
    sys.exit()


def log_timestamp(log_string):
    print(f"{get_progressed_time()}{log_string}")
