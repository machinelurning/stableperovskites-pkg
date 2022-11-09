from sklearn.pipeline import Pipeline

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset
from regression_model.processing.features import (
    CompositionAveragedProperties,
    NumOfSites,
    PopulateElementalProp,
    PopulateMajorityIonProperties,
    ReplaceBlank,
    ShannonRadius,
)

elemental_properties = load_dataset(file_name=config.app_config.elemental_prop)
shan_rad = load_dataset(file_name=config.app_config.shan_rad)

a_shan_rad = shan_rad[config.app_config.a_shan_rad_cols].dropna()
b_shan_rad = shan_rad[config.app_config.b_shan_rad_cols].dropna()


def test_replace_blank(sample_input_data):
    # Given
    transformer = ReplaceBlank()

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert (
        subject[config.app_config.replace_blank_col].iloc[
            config.app_config.replace_blank_loc
        ]
        == 0
    )


def test_populate_elemental_prop(sample_input_data):
    # Given
    transformer = Pipeline(
        [
            ("replace_blank_w_zero", ReplaceBlank()),
            (
                "populate_elemental_prop",
                PopulateElementalProp(
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
        ]
    )

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert (
        subject[config.app_config.populate_elem_col].iloc[
            config.app_config.populate_elem_loc
        ]
        == 3.59
    )


def test_num_of_sites(sample_input_data):
    # Given
    transformer = Pipeline(
        [
            ("replace_blank", ReplaceBlank()),
            (
                "populate_elemental_prop",
                PopulateElementalProp(
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
            ("count_num_sites", NumOfSites(sites=config.model_config.site_names)),
        ]
    )

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert (
        subject[config.app_config.num_site_test_col].iloc[
            config.app_config.num_site_test_loc
        ]
        == 1
    )


def test_shannon_radius(sample_input_data):
    # Given
    transformer = Pipeline(
        [
            ("replace_blank", ReplaceBlank()),
            (
                "populate_elemental_prop",
                PopulateElementalProp(
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
            ("count_num_sites", NumOfSites(sites=config.model_config.site_names)),
            (
                "add_shannon_radius",
                ShannonRadius(
                    a_shan=a_shan_rad,
                    b_shan=b_shan_rad,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
        ]
    )

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert (
        subject[config.app_config.a_shannon_radius_col].iloc[
            config.app_config.a_shannon_radius_loc
        ]
        == 1.34
    )
    assert (
        subject[config.app_config.b_shannon_radius_col].iloc[
            config.app_config.b_shannon_radius_loc
        ]
        == 0.634
    )


def test_comp_averaged(sample_input_data):
    # Given
    transformer = Pipeline(
        [
            ("replace_blank", ReplaceBlank()),
            (
                "populate_elemental_prop",
                PopulateElementalProp(
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
            ("count_num_sites", NumOfSites(sites=config.model_config.site_names)),
            (
                "add_shannon_radius",
                ShannonRadius(
                    a_shan=a_shan_rad,
                    b_shan=b_shan_rad,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
            (
                "add_structural_parameters",
                CompositionAveragedProperties(
                    sites=config.model_config.site_names,
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                ),
            ),
        ]
    )

    # When
    subject = transformer.fit_transform(sample_input_data)
    expected = dict(
        zip(
            config.app_config.comp_avgd_cols,
            config.app_config.comp_avgd_expected_vals,
        )
    )

    # Then
    vals = list(
        subject[config.app_config.comp_avgd_cols]
        .iloc[config.app_config.comp_avgd_loc]
        .values
    )
    subject = dict(zip(config.app_config.comp_avgd_cols, vals))

    assert expected == subject


def test_populate_maj_ion(sample_input_data):
    # Given
    transformer = Pipeline(
        [
            ("replace_blank", ReplaceBlank()),
            (
                "populate_elemental_prop",
                PopulateElementalProp(
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
            ("count_num_sites", NumOfSites(sites=config.model_config.site_names)),
            (
                "add_shannon_radius",
                ShannonRadius(
                    a_shan=a_shan_rad,
                    b_shan=b_shan_rad,
                    prefixes=config.model_config.prefixes,
                    sites=config.model_config.site_names,
                ),
            ),
            (
                "add_structural_parameters",
                CompositionAveragedProperties(
                    sites=config.model_config.site_names,
                    ep=elemental_properties,
                    prefixes=config.model_config.prefixes,
                ),
            ),
            (
                "populate_majority_ion_properties",
                PopulateMajorityIonProperties(
                    ep=elemental_properties, cols=config.model_config.max_cols
                ),
            ),
        ]
    )

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert (
        subject[config.app_config.avg_maj_col].iloc[config.app_config.avg_maj_loc]
        == 4.0
    )

    assert (
        subject[config.app_config.diff_maj_col].iloc[config.app_config.diff_maj_loc]
        == 0.48
    )
