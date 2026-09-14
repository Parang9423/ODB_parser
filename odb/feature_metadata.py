from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Sequence, Tuple

from hierarchy_renderer import FastODBRenderer
from odb.feature_query import ODBVectorFeature


@dataclass(frozen=True)
class TopLevelRecord:
    """One top-level record using the same counting rules as feature_query.

    Surface body records (OB/OS/OC/OE/SE) belong to the S record and therefore
    do not receive their own record_index. Unknown/directive records are kept so
    diagnostics can expose vendor-specific metadata without guessing semantics.
    """

    record_index: int
    command: str
    raw_record: str
    tail_tokens: Tuple[str, ...]
    semicolon_suffix: Tuple[str, ...]


def _tail_tokens(command: str, tokens: Sequence[str]) -> Tuple[str, ...]:
    command = command.upper()
    # These are parser-core fields only. Anything after them is deliberately
    # preserved as opaque metadata until a real ODB sample proves its meaning.
    core = {"P": 6, "L": 7, "S": 2}.get(command, 1)
    return tuple(tokens[core:]) if len(tokens) > core else ()


def _semicolon_suffix(raw: str) -> Tuple[str, ...]:
    parts = raw.split(";")
    if len(parts) <= 1:
        return ()
    return tuple(part.strip() for part in parts[1:] if part.strip())


def iter_top_level_records(records: Sequence[str]) -> Iterator[TopLevelRecord]:
    """Iterate feature-file records exactly enough to resolve feature_id suffixes."""
    index = 0
    record_index = 0
    while index < len(records):
        raw = records[index].strip()
        index += 1
        if not raw or raw.startswith("#") or raw.startswith("$") or raw == "SE":
            continue

        tokens = raw.split()
        if not tokens:
            continue
        command = tokens[0].upper()
        record_index += 1
        yield TopLevelRecord(
            record_index=record_index,
            command=command,
            raw_record=raw,
            tail_tokens=_tail_tokens(command, tokens),
            semicolon_suffix=_semicolon_suffix(raw),
        )

        if command == "S":
            while index < len(records):
                surface_raw = records[index].strip()
                index += 1
                if surface_raw.split(maxsplit=1)[0].upper() == "SE" if surface_raw else False:
                    break


def find_top_level_record(records: Sequence[str], record_index: int) -> TopLevelRecord | None:
    for record in iter_top_level_records(records):
        if record.record_index == record_index:
            return record
    return None


def _feature_record_index(feature: ODBVectorFeature) -> int | None:
    try:
        return int(feature.feature_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def describe_feature_source(renderer: FastODBRenderer, feature: ODBVectorFeature) -> dict:
    """Return lossless source-record evidence for one extracted vector feature."""
    feature_file = renderer._step_dir(feature.step) / "layers" / feature.layer.lower() / "features"
    _, records = renderer._feature_data(feature_file)
    wanted_index = _feature_record_index(feature)
    source = find_top_level_record(records, wanted_index) if wanted_index is not None else None
    if source is None:
        return {
            "source_step": feature.step,
            "source_layer": feature.layer,
            "source_record_index": wanted_index,
            "raw_record": None,
            "tail_tokens": [],
            "semicolon_suffix": [],
            "note": "Source record could not be resolved; no attribute meaning was inferred.",
        }
    return {
        "source_step": feature.step,
        "source_layer": feature.layer,
        "source_record_index": source.record_index,
        "raw_record": source.raw_record,
        "tail_tokens": list(source.tail_tokens),
        "semicolon_suffix": list(source.semicolon_suffix),
        "note": "Tail/semicolon fields are preserved as opaque metadata until their syntax is verified on this ODB dataset.",
    }


def scan_feature_file_metadata(records: Sequence[str], sample_limit: int = 30) -> dict:
    """Inventory metadata-like syntax without assigning ODB business semantics.

    The diagnostic intentionally records prefixes and unknown top-level commands
    rather than assuming that a vendor's @/&/!/etc. syntax means net, pad or
    component. Geometry-record tails are also sampled because attributes may be
    encoded inline.
    """
    prefix_counts: Counter[str] = Counter()
    prefix_samples: Dict[str, list[str]] = {}
    unknown_commands: Counter[str] = Counter()
    unknown_samples: Dict[str, list[str]] = {}
    geometry_tail_count = 0
    geometry_tail_samples: list[dict] = []

    for raw_value in records:
        raw = raw_value.strip()
        if not raw or raw.startswith("#") or raw.startswith("$"):
            continue
        if raw[0] in "@&%!":
            prefix = raw[0]
            prefix_counts[prefix] += 1
            bucket = prefix_samples.setdefault(prefix, [])
            if len(bucket) < sample_limit:
                bucket.append(raw)

    for record in iter_top_level_records(records):
        if record.command in {"P", "L", "S"}:
            if record.tail_tokens or record.semicolon_suffix:
                geometry_tail_count += 1
                if len(geometry_tail_samples) < sample_limit:
                    geometry_tail_samples.append({
                        "record_index": record.record_index,
                        "command": record.command,
                        "raw_record": record.raw_record,
                        "tail_tokens": list(record.tail_tokens),
                        "semicolon_suffix": list(record.semicolon_suffix),
                    })
            continue
        unknown_commands[record.command] += 1
        bucket = unknown_samples.setdefault(record.command, [])
        if len(bucket) < sample_limit:
            bucket.append(record.raw_record)

    return {
        "metadata_prefix_counts": dict(sorted(prefix_counts.items())),
        "metadata_prefix_samples": dict(sorted(prefix_samples.items())),
        "unknown_top_level_command_counts": dict(sorted(unknown_commands.items())),
        "unknown_top_level_command_samples": dict(sorted(unknown_samples.items())),
        "geometry_records_with_extra_fields": geometry_tail_count,
        "geometry_extra_field_samples": geometry_tail_samples,
        "interpretation_note": (
            "This is a syntax inventory only. Do not map any prefix/token to PAD/CIRCUIT/NET/COMPONENT "
            "until the emitted raw records are interpreted against this dataset."
        ),
    }


def summarize_feature_sources(
    renderer: FastODBRenderer,
    features: Iterable[ODBVectorFeature],
    sample_limit: int = 30,
) -> list[dict]:
    """Scan each unique STEP/layer source file once, despite repeated instances."""
    unique = sorted({(feature.step.lower(), feature.layer.lower()) for feature in features})
    summaries: list[dict] = []
    for step, layer in unique:
        feature_file = renderer._step_dir(step) / "layers" / layer / "features"
        _, records = renderer._feature_data(feature_file)
        payload = scan_feature_file_metadata(records, sample_limit=sample_limit)
        payload.update({
            "step": step,
            "layer": layer,
            "record_line_count": len(records),
        })
        summaries.append(payload)
    return summaries
