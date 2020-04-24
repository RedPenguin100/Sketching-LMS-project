import time


class MeasurementReturn:
    def __init__(self, return_value, duration):
        self.return_value = return_value
        self.duration = duration


def measure_method(method, *args, **kwargs):
    start = time.time()
    return_value = method(*args, **kwargs)
    end = time.time()
    return MeasurementReturn(return_value, end - start)
