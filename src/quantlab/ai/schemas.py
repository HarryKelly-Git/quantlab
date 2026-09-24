"""Structured-output contracts for the AI roles.

Two representations of the same contract live here on purpose:

* **Canonical JSON Schemas** (hand-written, deterministic) are sent to providers. Each provider
  supports a different subset of JSON Schema (Anthropic rejects numeric/length bounds, OpenAI
  strict mode needs every property required and ``additionalProperties: false``, Gemini silently
  drops keywords it does not know), so :func:`sanitize_schema` rewrites the canonical schema per
  provider. Because providers may therefore enforce LESS than the canonical schema, the provider
  layer ALWAYS re-validates the reply client-side with :func:`validate_instance`.
* **Pydantic models** turn a validated reply into typed objects, normalize enum casing (Anthropic
  documents that enum capitalization is not guaranteed) and enforce the same bounds again.

A test asserts that both representations agree (same properties, enums and limits).
"""
from __future__ import annotations

import copy
import json
import re
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from quantlab.core.types import AIDecision, ObjectionSeverity

# Client-side bounds. Providers cannot all enforce these (see module docstring), so pydantic and
# validate_instance() enforce them after every call.
MAX_TEXT = 2000          # one claim / objection / list entry
MAX_LONG_TEXT = 6000     # summary / rationale
MAX_ITEMS = 25           # any list
MAX_PATH = 200           # source_field path
MAX_TITLE = 200
MAX_IDEAS = 10


class Confidence(str, Enum):
    """Qualitative label ONLY. Never a probability (ARCHITECTURE.md section 7)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ObjectionCategory(str, Enum):
    THESIS_WEAKNESS = "thesis_weakness"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    DATA_QUALITY = "data_quality"
    VALUATION = "valuation"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    EVENT_RISK = "event_risk"
    LIQUIDITY = "liquidity"
    REGIME_CONFLICT = "regime_conflict"
    SECTOR_WEAKNESS = "sector_weakness"
    CROWDING = "crowding"
    TEMPORARY_NEWS = "temporary_news"
    MISLEADING_CORRELATION = "misleading_correlation"
    OVERFITTING = "overfitting"
    STRATEGY_WEAKNESS = "strategy_weakness"
    OTHER = "other"


DECISION_VALUES = [d.value for d in AIDecision]
SEVERITY_VALUES = [s.value for s in ObjectionSeverity]
CONFIDENCE_VALUES = [c.value for c in Confidence]
CATEGORY_VALUES = [c.value for c in ObjectionCategory]


class OutputValidationError(ValueError):
    """A provider reply that is valid JSON but does not satisfy the role contract."""


# --------------------------------------------------------------------------------------------
# Pydantic models (typed view of a validated reply)
# --------------------------------------------------------------------------------------------
Text = Annotated[str, Field(max_length=MAX_TEXT)]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT)]
LongText = Annotated[str, Field(max_length=MAX_LONG_TEXT)]
SourcePath = Annotated[str, Field(min_length=1, max_length=MAX_PATH)]


def _upper(v: Any) -> Any:
    return v.strip().upper() if isinstance(v, str) else v


def _lower(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


class _Strict(BaseModel):
    # extra="forbid": a reply carrying keys we did not ask for (e.g. an invented "probability" or
    # "status": "SUPPORTED") is rejected rather than silently ignored.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceItem(_Strict):
    claim: NonEmptyText
    source_field: SourcePath


class ResearcherOutput(_Strict):
    summary: LongText
    key_evidence: list[EvidenceItem] = Field(max_length=MAX_ITEMS)
    missing_information: list[Text] = Field(max_length=MAX_ITEMS)
    alternative_explanations: list[Text] = Field(max_length=MAX_ITEMS)
    decision: AIDecision
    qualitative_confidence: Confidence

    @field_validator("decision", mode="before")
    @classmethod
    def _norm_decision(cls, v: Any) -> Any:
        return _upper(v)

    @field_validator("qualitative_confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v: Any) -> Any:
        return _lower(v)


class ObjectionItem(_Strict):
    category: ObjectionCategory
    severity: ObjectionSeverity
    text: NonEmptyText
    source_field: SourcePath

    @field_validator("category", mode="before")
    @classmethod
    def _norm_cat(cls, v: Any) -> Any:
        return _lower(v)

    @field_validator("severity", mode="before")
    @classmethod
    def _norm_sev(cls, v: Any) -> Any:
        return _upper(v)


class AdversaryOutput(_Strict):
    objections: list[ObjectionItem] = Field(max_length=MAX_ITEMS)
    overall_decision: AIDecision

    @field_validator("overall_decision", mode="before")
    @classmethod
    def _norm_decision(cls, v: Any) -> Any:
        return _upper(v)


class DecisiveFactor(_Strict):
    factor: NonEmptyText
    source_field: SourcePath


class JudgeOutput(_Strict):
    decision: AIDecision
    rationale: LongText
    decisive_factors: list[DecisiveFactor] = Field(max_length=MAX_ITEMS)
    considered_objections: list[Text] = Field(max_length=MAX_ITEMS)
    unknowns: list[Text] = Field(max_length=MAX_ITEMS)

    @field_validator("decision", mode="before")
    @classmethod
    def _norm_decision(cls, v: Any) -> Any:
        return _upper(v)


class IdeaItem(_Strict):
    title: Annotated[str, Field(min_length=1, max_length=MAX_TITLE)]
    hypothesis: NonEmptyText
    rationale: NonEmptyText
    required_data: NonEmptyText
    proposed_test: NonEmptyText
    expected_failure_conditions: NonEmptyText


class IdeaBatch(_Strict):
    ideas: list[IdeaItem] = Field(max_length=MAX_IDEAS)


# --------------------------------------------------------------------------------------------
# Canonical JSON Schemas (what providers are asked to follow)
# --------------------------------------------------------------------------------------------
def _s(description: str, max_length: int = MAX_TEXT, min_length: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "string", "description": description, "maxLength": max_length}
    if min_length is not None:
        out["minLength"] = min_length
    return out


def _enum(values: list[str], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values), "description": description}


def _arr(items: dict[str, Any], description: str, max_items: int = MAX_ITEMS) -> dict[str, Any]:
    return {"type": "array", "items": items, "description": description, "maxItems": max_items}


def _obj(props: dict[str, Any], title: str | None = None, description: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "object"}
    if title:
        out["title"] = title
    if description:
        out["description"] = description
    out["properties"] = props
    out["required"] = list(props)
    out["additionalProperties"] = False
    return out


_SOURCE_FIELD = _s("Dotted path of the evidence-packet field supporting this, e.g. 'features.ret_20d' "
                   "or 'no_trade_checks[0].passed'. Must exist in the packet.", MAX_PATH, 1)

RESEARCHER_SCHEMA: dict[str, Any] = _obj({
    "summary": _s("Neutral summary of the evidence. No new numbers.", MAX_LONG_TEXT),
    "key_evidence": _arr(_obj({
        "claim": _s("One claim taken from the packet.", MAX_TEXT, 1),
        "source_field": _SOURCE_FIELD,
    }), "Evidence items, each citing a packet field."),
    "missing_information": _arr(_s("Information that is UNKNOWN / absent from the packet."), "Missing information."),
    "alternative_explanations": _arr(_s("A different explanation of the signal."), "Alternative explanations."),
    "decision": _enum(DECISION_VALUES, "ACCEPT, REJECT, WATCH or UNKNOWN."),
    "qualitative_confidence": _enum(CONFIDENCE_VALUES, "Qualitative label, NOT a probability."),
}, title="ResearcherOutput")

ADVERSARY_SCHEMA: dict[str, Any] = _obj({
    "objections": _arr(_obj({
        "category": _enum(CATEGORY_VALUES, "Objection category."),
        "severity": _enum(SEVERITY_VALUES, "HARD_FAIL, MATERIAL_CONCERN, MINOR_CONCERN or UNKNOWN."),
        "text": _s("The objection, grounded in the packet.", MAX_TEXT, 1),
        "source_field": _SOURCE_FIELD,
    }), "Reasons NOT to take this trade."),
    "overall_decision": _enum(DECISION_VALUES, "ACCEPT, REJECT, WATCH or UNKNOWN."),
}, title="AdversaryOutput")

JUDGE_SCHEMA: dict[str, Any] = _obj({
    "decision": _enum(DECISION_VALUES, "ACCEPT, REJECT, WATCH or UNKNOWN."),
    "rationale": _s("Why, citing packet fields. No new numbers.", MAX_LONG_TEXT),
    "decisive_factors": _arr(_obj({
        "factor": _s("A decisive factor from the packet.", MAX_TEXT, 1),
        "source_field": _SOURCE_FIELD,
    }), "Decisive factors, each citing a packet field."),
    "considered_objections": _arr(_s("An adversary objection and how it was weighed."), "Objections considered."),
    "unknowns": _arr(_s("Something that remains UNKNOWN."), "Unknowns."),
}, title="JudgeOutput")

IDEAS_SCHEMA: dict[str, Any] = _obj({
    "ideas": _arr(_obj({
        "title": _s("Short title.", MAX_TITLE, 1),
        "hypothesis": _s("A falsifiable hypothesis.", MAX_TEXT, 1),
        "rationale": _s("Why, using only the supplied observations.", MAX_TEXT, 1),
        "required_data": _s("Data needed to test it.", MAX_TEXT, 1),
        "proposed_test": _s("How to test it out-of-sample.", MAX_TEXT, 1),
        "expected_failure_conditions": _s("What result would falsify it.", MAX_TEXT, 1),
    }), "Proposed research hypotheses (untested).", MAX_IDEAS),
}, title="IdeaBatch")

ROLE_CONTRACTS: dict[str, tuple[type[_Strict], dict[str, Any]]] = {
    "researcher": (ResearcherOutput, RESEARCHER_SCHEMA),
    "adversary": (AdversaryOutput, ADVERSARY_SCHEMA),
    "judge": (JudgeOutput, JUDGE_SCHEMA),
    "idea_generator": (IdeaBatch, IDEAS_SCHEMA),
}


def schema_for(role: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(ROLE_CONTRACTS[role][1])
    except KeyError as exc:
        raise KeyError(f"unknown AI role {role!r}") from exc


def schema_name(role: str) -> str:
    return str(ROLE_CONTRACTS[role][1].get("title", role))


def parse_output(role: str, data: Any) -> _Strict:
    """Validate a decoded reply against the role's pydantic model (raises OutputValidationError)."""
    model_cls = ROLE_CONTRACTS[role][0]
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        # Keep the message short; pydantic errors can echo large inputs.
        raise OutputValidationError(f"{role} output invalid: {exc.error_count()} error(s): "
                                    f"{_short_errors(exc)}") from None


def _short_errors(exc: ValidationError, limit: int = 5) -> str:
    parts = []
    for e in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"{loc}: {e.get('type')}")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------------
# Client-side JSON Schema validation (subset used by our canonical schemas)
# --------------------------------------------------------------------------------------------
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_instance(data: Any, schema: dict[str, Any], path: str = "$", max_errors: int = 20) -> list[str]:
    """Validate ``data`` against a JSON Schema subset. Returns a list of error strings (empty = valid).

    Supports type (incl. type arrays), enum (string enums compared CASE-INSENSITIVELY because
    providers do not guarantee casing), const, properties/required/additionalProperties:false,
    items, anyOf, min/maxItems, min/maxLength, minimum/maximum. Unknown keywords are ignored, which
    is safe because the canonical schemas only use the supported ones (tested).
    """
    errors: list[str] = []
    _validate(data, schema, path, errors)
    return errors[:max_errors]


def _validate(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if "anyOf" in schema:
        if not any(not validate_instance(data, sub, path) for sub in schema["anyOf"]):
            errors.append(f"{path}: matches no anyOf alternative")
        return
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPE_CHECKS.get(x, lambda _v: False)(data) for x in types):
            errors.append(f"{path}: expected {'/'.join(types)}, got {type(data).__name__}")
            return
    if "enum" in schema:
        allowed = schema["enum"]
        if isinstance(data, str):
            ok = any(isinstance(a, str) and a.lower() == data.strip().lower() for a in allowed)
        else:
            ok = data in allowed
        if not ok:
            errors.append(f"{path}: value not in enum")
    if "const" in schema and data != schema["const"]:
        errors.append(f"{path}: value != const")
    if isinstance(data, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, sub in props.items():
            if key in data:
                _validate(data[key], sub, f"{path}.{key}", errors)
    elif isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(data):
                _validate(item, schema["items"], f"{path}[{i}]", errors)
    elif isinstance(data, str):
        if "minLength" in schema and len(data.strip()) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: above maximum")


def parse_json_text(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Decode a model's text reply into a JSON object. Returns (data, error)."""
    if text is None or not text.strip():
        return None, "empty response text"
    s = text.strip()
    # Tolerate a single surrounding markdown fence (some non-strict modes add one); nothing else.
    fence = re.match(r"^```(?:json)?\s*\n(.*)\n```$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid JSON: {exc.msg if hasattr(exc, 'msg') else exc}"
    if not isinstance(data, dict):
        return None, f"JSON root must be an object, got {type(data).__name__}"
    return data, None


# --------------------------------------------------------------------------------------------
# Provider-specific schema sanitizing
# --------------------------------------------------------------------------------------------
_ANNOTATION_KEYS = {"title", "$schema", "$id", "examples", "$comment"}
# Anthropic structured outputs reject numeric bounds, length bounds and most array constraints (400).
_ANTHROPIC_DROP = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                   "minLength", "maxLength", "maxItems", "uniqueItems", "minProperties",
                   "maxProperties", "contains", "minContains", "maxContains"}
# OpenAI strict mode: unsupported composition keywords and string length bounds.
_OPENAI_DROP = {"allOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas",
                "minLength", "maxLength", "uniqueItems", "minProperties", "maxProperties"}
# Gemini: keep only documented keywords; everything else would be silently ignored anyway.
_GEMINI_KEEP = {"type", "properties", "required", "additionalProperties", "enum", "format", "items",
                "prefixItems", "minItems", "maxItems", "minimum", "maximum", "description", "anyOf",
                "propertyOrdering", "$defs", "$ref"}


def sanitize_schema(schema: dict[str, Any], provider: str) -> dict[str, Any]:
    """Rewrite a canonical schema for ``provider`` (anthropic | openai | google | mock)."""
    if provider == "anthropic":
        return _walk(schema, _anthropic_node)
    if provider == "openai":
        return _walk(schema, _openai_node)
    if provider == "google":
        return _walk(schema, _gemini_node)
    return copy.deepcopy(schema)


def _walk(schema: Any, fn) -> Any:
    if isinstance(schema, list):
        return [_walk(s, fn) for s in schema]
    if not isinstance(schema, dict):
        return schema
    node = fn({k: v for k, v in schema.items()})
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _walk(pv, fn) for pk, pv in v.items()}
        elif k in ("items", "additionalProperties") and isinstance(v, dict):
            out[k] = _walk(v, fn)
        elif k in ("anyOf", "oneOf", "prefixItems") and isinstance(v, list):
            out[k] = [_walk(s, fn) for s in v]
        elif k == "$defs" and isinstance(v, dict):
            out[k] = {dk: _walk(dv, fn) for dk, dv in v.items()}
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_object(node: dict[str, Any]) -> bool:
    t = node.get("type")
    return t == "object" or (isinstance(t, list) and "object" in t) or "properties" in node


def _anthropic_node(node: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in node.items() if k not in _ANTHROPIC_DROP and k not in _ANNOTATION_KEYS}
    if "minItems" in out and out["minItems"] not in (0, 1):
        out.pop("minItems")        # only 0 or 1 accepted
    if _is_object(out):
        out["additionalProperties"] = False
    return out


def _nullable(sub: dict[str, Any]) -> dict[str, Any]:
    sub = dict(sub)
    t = sub.get("type")
    if isinstance(t, str):
        sub["type"] = [t, "null"]
    elif isinstance(t, list):
        if "null" not in t:
            sub["type"] = [*t, "null"]
    else:
        sub = {"anyOf": [sub, {"type": "null"}]}
    if "enum" in sub and None not in sub["enum"]:
        sub["enum"] = [*sub["enum"], None]
    return sub


def _openai_node(node: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in node.items() if k not in _OPENAI_DROP and k not in _ANNOTATION_KEYS}
    if "oneOf" in out:
        out["anyOf"] = out.pop("oneOf")
    if _is_object(out):
        props = dict(out.get("properties", {}))
        required = set(out.get("required", []))
        for key in list(props):
            if key not in required:
                props[key] = _nullable(props[key])      # optional -> required + nullable
        out["properties"] = props
        out["required"] = list(props)
        out["additionalProperties"] = False
    return out


def _gemini_node(node: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in node.items() if k in _GEMINI_KEEP}
    if "oneOf" in node:
        out["anyOf"] = node["oneOf"]
    if _is_object(out) and "properties" in out:
        out["propertyOrdering"] = list(out["properties"])
    return out
