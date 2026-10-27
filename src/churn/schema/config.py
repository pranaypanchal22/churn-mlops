"""Typed loaders for the canonical schema and source mapping configs.

Config errors (a mapping that references an unknown field, maps a forbidden column, or
produces tokens the schema does not allow) are raised at load time, before any data is read.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigError(ValueError):
    """A schema or mapping file is inconsistent."""


class _Strict(BaseModel):
    # Typos in config keys should fail loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetSpec(_Strict):
    name: str
    positive_value: str
    negative_value: str
    description: str = ""


class FieldSpec(_Strict):
    name: str
    dtype: Literal["integer", "float", "category"]
    role: Literal["feature", "audit"] = "feature"
    required: bool = True
    allowed_values: list[str] | None = None
    min: float | None = None
    max: float | None = None
    max_missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str = ""

    @model_validator(mode="after")
    def _check_type_constraints(self) -> FieldSpec:
        if self.dtype == "category" and not self.allowed_values:
            raise ValueError(f"{self.name}: category fields need allowed_values")
        if self.dtype != "category" and self.allowed_values:
            raise ValueError(f"{self.name}: allowed_values only applies to category fields")
        return self

    @property
    def is_numeric(self) -> bool:
        return self.dtype in ("integer", "float")


class CanonicalSchema(_Strict):
    schema_version: str
    target: TargetSpec
    fields: list[FieldSpec]
    forbidden_source_patterns: list[str] = []

    @model_validator(mode="after")
    def _unique_names(self) -> CanonicalSchema:
        names = [f.name for f in self.fields] + [self.target.name]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate field names in schema: {sorted(dupes)}")
        return self

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.fields if f.role == "feature"]

    def is_forbidden_source(self, column: str) -> bool:
        return any(re.search(p, column, re.IGNORECASE) for p in self.forbidden_source_patterns)


class FillRule(_Strict):
    field: str
    value: float | str
    when_field: str
    when_equals: float | str


class CleaningSpec(_Strict):
    strip_whitespace: bool = True
    blank_as_missing: bool = True
    fill_rules: list[FillRule] = []


class SourceMapping(_Strict):
    mapping_name: str
    mapping_version: str
    schema_version: str
    description: str = ""
    unexpected_columns: Literal["error", "warn"] = "error"
    columns: dict[str, list[str]]
    identifier_columns: list[str] = []
    excluded_columns: list[str] = []
    ignored_columns: list[str] = []
    value_maps: dict[str, dict[str, str]] = {}
    cleaning: CleaningSpec = CleaningSpec()

    @property
    def field_value_maps(self) -> dict[str, dict[str, str]]:
        # Underscore-prefixed entries exist only to hold YAML anchors.
        return {k: v for k, v in self.value_maps.items() if not k.startswith("_")}

    @property
    def declared_non_feature_columns(self) -> set[str]:
        return set(self.identifier_columns) | set(self.excluded_columns) | set(self.ignored_columns)


def _read_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def load_schema(path: Path) -> CanonicalSchema:
    return CanonicalSchema.model_validate(_read_yaml(path))


def check_mapping_against_schema(mapping: SourceMapping, schema: CanonicalSchema) -> None:
    """Raise ConfigError listing every inconsistency between a mapping and the schema."""
    problems: list[str] = []
    known = set(schema.field_names) | {schema.target.name}

    if mapping.schema_version != schema.schema_version:
        problems.append(
            f"mapping targets schema {mapping.schema_version}, "
            f"but loaded schema is {schema.schema_version}"
        )

    for name in sorted(set(mapping.columns) - known):
        problems.append(f"columns: '{name}' is not a canonical field")
    for name in sorted(set(mapping.field_value_maps) - known):
        problems.append(f"value_maps: '{name}' is not a canonical field")

    if schema.target.name not in mapping.columns:
        problems.append(f"columns: target '{schema.target.name}' is not mapped")
    for f in schema.fields:
        if f.required and f.name not in mapping.columns:
            problems.append(f"columns: required field '{f.name}' is not mapped")

    # One source column must not feed two canonical fields, and declared non-feature
    # columns must not also be mapped.
    seen: dict[str, str] = {}
    for canonical, aliases in mapping.columns.items():
        if not aliases:
            problems.append(f"columns: '{canonical}' has no source aliases")
        for alias in aliases:
            if alias in seen and seen[alias] != canonical:
                problems.append(
                    f"columns: source '{alias}' is mapped to both '{seen[alias]}' and '{canonical}'"
                )
            seen[alias] = canonical
            if alias in mapping.declared_non_feature_columns:
                problems.append(
                    f"columns: source '{alias}' is mapped but also declared as "
                    "identifier/excluded/ignored"
                )
            if canonical != schema.target.name and schema.is_forbidden_source(alias):
                problems.append(
                    f"columns: source '{alias}' matches a forbidden pattern (identifier or "
                    f"target-derived) and cannot feed '{canonical}'"
                )

    for name, vmap in mapping.field_value_maps.items():
        if name == schema.target.name:
            allowed = {schema.target.positive_value, schema.target.negative_value}
        elif name in schema.field_names:
            spec = schema.field(name)
            if spec.is_numeric:
                problems.append(f"value_maps: '{name}' is numeric; value maps apply to categories")
                continue
            allowed = set(spec.allowed_values or [])
        else:
            continue
        bad = sorted(set(vmap.values()) - allowed)
        if bad:
            problems.append(f"value_maps: '{name}' maps to tokens not in schema: {bad}")

    for rule in mapping.cleaning.fill_rules:
        for ref in (rule.field, rule.when_field):
            if ref not in schema.field_names:
                problems.append(f"cleaning.fill_rules: '{ref}' is not a canonical field")

    if problems:
        raise ConfigError(
            f"mapping '{mapping.mapping_name}' v{mapping.mapping_version} is invalid:\n  - "
            + "\n  - ".join(problems)
        )


def load_mapping(path: Path, schema: CanonicalSchema) -> SourceMapping:
    mapping = SourceMapping.model_validate(_read_yaml(path))
    check_mapping_against_schema(mapping, schema)
    return mapping
