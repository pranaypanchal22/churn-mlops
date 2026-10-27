import pytest
from pydantic import ValidationError

from churn.schema.config import ConfigError, SourceMapping, check_mapping_against_schema


def _with(mapping: SourceMapping, **update) -> SourceMapping:
    return mapping.model_copy(update=update)


def test_shipped_configs_are_consistent(schema, kaggle_mapping, ibm_mapping):
    assert kaggle_mapping.schema_version == schema.schema_version
    assert ibm_mapping.schema_version == schema.schema_version


def test_audit_field_is_not_a_model_feature(schema):
    assert "gender" in schema.field_names
    assert "gender" not in schema.feature_names
    assert schema.target.name not in schema.feature_names


def test_unknown_canonical_field_rejected(schema, kaggle_mapping):
    bad = _with(kaggle_mapping, columns={**kaggle_mapping.columns, "favourite_colour": ["X"]})
    with pytest.raises(ConfigError, match="favourite_colour"):
        check_mapping_against_schema(bad, schema)


def test_required_field_must_be_mapped(schema, kaggle_mapping):
    cols = {k: v for k, v in kaggle_mapping.columns.items() if k != "contract"}
    with pytest.raises(ConfigError, match="required field 'contract' is not mapped"):
        check_mapping_against_schema(_with(kaggle_mapping, columns=cols), schema)


@pytest.mark.parametrize("source", ["Churn Score", "Customer ID", "CLTV", "Churn Reason"])
def test_forbidden_source_cannot_feed_a_feature(schema, kaggle_mapping, source):
    cols = {**kaggle_mapping.columns, "tenure_months": [source]}
    with pytest.raises(ConfigError, match="forbidden pattern"):
        check_mapping_against_schema(_with(kaggle_mapping, columns=cols), schema)


def test_one_source_column_cannot_feed_two_fields(schema, kaggle_mapping):
    cols = {**kaggle_mapping.columns, "total_charges": ["Monthly Charges"]}
    with pytest.raises(ConfigError, match="mapped to both"):
        check_mapping_against_schema(_with(kaggle_mapping, columns=cols), schema)


def test_value_map_tokens_must_be_allowed_by_schema(schema, kaggle_mapping):
    vmaps = {**kaggle_mapping.value_maps, "contract": {"Monthly": "monthly"}}
    with pytest.raises(ConfigError, match="tokens not in schema"):
        check_mapping_against_schema(_with(kaggle_mapping, value_maps=vmaps), schema)


def test_schema_version_mismatch_rejected(schema, kaggle_mapping):
    with pytest.raises(ConfigError, match="targets schema 9.9.9"):
        check_mapping_against_schema(_with(kaggle_mapping, schema_version="9.9.9"), schema)


def test_unknown_config_keys_rejected():
    with pytest.raises(ValidationError):
        SourceMapping.model_validate(
            {
                "mapping_name": "x",
                "mapping_version": "1",
                "schema_version": "1.0.0",
                "columns": {},
                "colums_typo": {},
            }
        )
