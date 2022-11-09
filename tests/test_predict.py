from regression_model.config.core import config
from regression_model.predict import make_prediction
from regression_model.processing.data_manager import load_dataset


def test_make_prediction_df():
    df = load_dataset(file_name=config.app_config.test_split)
    results = make_prediction(input_data=df)
    assert results["errors"] is None
