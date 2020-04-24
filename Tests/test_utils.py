import pytest
import time

from pytest import approx
from Source.Utils.measurement import measure_method


@pytest.mark.parametrize('seconds_to_sleep', [0.1, 0.2, 1, 0.5])
def test_measurement_function(seconds_to_sleep):
    # 0.01 seconds room for error
    err = 1e-2
    assert approx(measure_method(time.sleep, seconds_to_sleep).duration, abs=err) == seconds_to_sleep
