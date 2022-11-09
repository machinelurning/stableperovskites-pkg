import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.app_config.dft_calc_test)


@pytest.fixture()
def sample_input_data_validation():
    return load_dataset(file_name=config.app_config.dft_calc_test_validation)
