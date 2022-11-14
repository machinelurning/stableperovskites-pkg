from regression_model.config.core import config
from regression_model.processing.validation import validate_inputs


def test_validate_nulls_in_required(sample_input_data_validation):
    validated_data, error = validate_inputs(
        input_data=sample_input_data_validation.iloc[
            config.app_config.no_nulls_in_required_loc
        ]
        .to_frame()
        .T
    )
    assert validated_data is None
    assert error is not None


def test_validate_a_site_not_present(sample_input_data_validation):

    validated_data, error = validate_inputs(
        input_data=sample_input_data_validation.iloc[
            config.app_config.a_site_not_present_loc
        ]
        .to_frame()
        .T
    )

    assert validated_data is None
    assert error is not None


def test_validate_b_site_not_present(sample_input_data_validation):
    validated_data, error = validate_inputs(
        input_data=sample_input_data_validation.iloc[
            config.app_config.b_site_not_present_loc
        ]
        .to_frame()
        .T
    )

    assert validated_data is None
    assert error is not None


def test_validate_non_stochiometric(sample_input_data_validation):
    validated_data, error = validate_inputs(
        input_data=sample_input_data_validation.iloc[
            config.app_config.non_stoichiometric_loc
        ]
        .to_frame()
        .T
    )
    assert validated_data is None
    assert error is not None
