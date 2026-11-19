"""Build and audit the shared, pre-registered master stimulus config.

Run from the repository root with the eas-gpu environment. The script downloads
or reads only the pinned Pythia tokenizer; it does not load model weights or run
an attention experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
UNICODE_DATA = ROOT / "data" / "unicode" / "emoji-test-15.1.txt"
DEFAULT_OUTPUT = ROOT / "config" / "master_stimuli.yaml"

TARGET_A_PER_BLOCK = 12
TARGET_B_PER_RANGE = [10, 6, 4]
TARGET_C = 20
TARGET_D = 15

HEX_PATTERN = "3a7f1c9e5b2d8a4f6c0e7b3d"
CONTROL_MAX_HEX_LENGTH = 24


A_BLOCKS = [
    (
        "U+1F300-U+1F5FF",
        0x1F300,
        0x1F5FF,
        [
            "1F300", "1F308", "1F30A", "1F30B", "1F30D", "1F30E",
            "1F319", "1F31F", "1F320", "1F34E", "1F355", "1F389",
        ],
    ),
    (
        "U+1F600-U+1F64F",
        0x1F600,
        0x1F64F,
        [
            "1F600", "1F603", "1F604", "1F601", "1F606", "1F605",
            "1F602", "1F643", "1F609", "1F60A", "1F60D", "1F642",
        ],
    ),
    (
        "U+1F680-U+1F6FF",
        0x1F680,
        0x1F6FF,
        [
            "1F680", "1F681", "1F682", "1F684", "1F685", "1F686",
            "1F687", "1F68A", "1F68C", "1F691", "1F695", "1F6B2",
        ],
    ),
    (
        "U+1F900-U+1F9FF",
        0x1F900,
        0x1F9FF,
        [
            "1F914", "1F917", "1F920", "1F921", "1F922", "1F927",
            "1F92F", "1F929", "1F92B", "1F92D", "1F916", "1F9E0",
        ],
    ),
    (
        "U+1FA70-U+1FAFF",
        0x1FA70,
        0x1FAFF,
        [
            "1FA70", "1FA71", "1FA72", "1FA73", "1FA74", "1FA80",
            "1FA81", "1FA82", "1FA84", "1FA85", "1FAA9", "1FA86",
        ],
    ),
]


A_GLOSSES = {
    "1F30A": "wave",
    "1F319": "moon",
    "1F31F": "star",
    "1F34E": "apple",
    "1F355": "pizza",
    "1F389": "party",
    "1F600": "grin",
    "1F602": "laugh",
    "1F60A": "smile",
    "1F60D": "heart",
    "1F680": "rocket",
    "1F681": "helicopter",
    "1F6B2": "bicycle",
    "1F916": "robot",
    "1F9E0": "brain",
}


B_RANGES = [
    (
        "U+2600-U+26FF",
        0x2600,
        0x26FF,
        10,
        [
            ("2600", "FE0F"), ("2601", "FE0F"), ("2602", "FE0F"),
            ("2603",), ("2604", "FE0F"), ("260E", "FE0F"),
            ("2614",), ("2615",), ("2648",), ("2649",), ("264A",),
            ("264B",), ("264C",), ("264D",), ("264E",), ("264F",),
            ("2650",), ("2651",), ("2652",), ("2653",), ("267B", "FE0F"),
            ("267F",), ("2693",), ("26A1",), ("26AA",), ("26AB",),
        ],
    ),
    (
        "U+2700-U+27BF",
        0x2700,
        0x27BF,
        6,
        [
            ("2702", "FE0F"), ("2708", "FE0F"), ("2709", "FE0F"),
            ("2714", "FE0F"), ("2716", "FE0F"), ("2728",),
            ("2744", "FE0F"), ("2764", "FE0F"), ("2795",),
            ("2796",), ("2797",), ("27A1", "FE0F"),
        ],
    ),
    (
        "U+2B00-U+2BFF",
        0x2B00,
        0x2BFF,
        4,
        [
            ("2B1B", "FE0F"), ("2B1C", "FE0F"), ("2B50",), ("2B55",),
            ("2B06", "FE0F"), ("2B07", "FE0F"), ("2B05", "FE0F"),
        ],
    ),
]


C_CANDIDATES = [
    (["1F44D", "1F3FB"], "thumbs up with light skin tone", 6),
    (["1F44D", "1F3FD"], "thumbs up with medium skin tone", 6),
    (["1F44B", "1F3FB"], "wave with light skin tone", 6),
    (["1F44B", "1F3FF"], "wave with dark skin tone", 6),
    (["1F44F", "1F3FB"], "clapping with light skin tone", 6),
    (["1F64F", "1F3FD"], "folded hands with medium skin tone", 6),
    (["1F4AA", "1F3FD"], "flexed arm with medium skin tone", 6),
    (["1F44E", "1F3FE"], "thumbs down with medium dark skin tone", 6),
    (["1F9D1", "200D", "1F4BB"], "computer user sequence", 8),
    (["1F469", "200D", "1F4BB"], "laptop user sequence A", 8),
    (["1F469", "200D", "1F52C"], "lab instrument user sequence A", 8),
    (["1F469", "200D", "1F680"], "space traveler sequence", 8),
    (["1F468", "200D", "1F469", "200D", "1F467"], "three person family sequence", 13),
    (["1F468", "200D", "1F469", "200D", "1F467", "200D", "1F466"], "four person family sequence", 18),
    (["1F1F8", "1F1EC"], "Singapore flag", 6),
    (["1F1FA", "1F1F8"], "United States flag", 6),
    (["1F1EC", "1F1E7"], "United Kingdom flag", 6),
    (["1F1EF", "1F1F5"], "Japan flag", 6),
    (["1F1E8", "1F1E6"], "Canada flag", 6),
    (["1F1EB", "1F1F7"], "France flag", 6),
]


D_CANDIDATE_CODEPOINTS = [
    "2122", "2116", "2103", "2030", "2020", "2021", "2605", "2192",
    "2194", "2713", "2026", "2032", "2033", "2044", "20AC", "2212",
    "2217", "2236", "223C", "2264", "2265", "22EF", "2500", "2550",
    "2588", "25A0", "25AA", "25CF", "266A", "2010", "2011", "2013",
    "2014", "2015", "2018", "2019", "201C", "201D", "201F", "2022",
]


REFERENCE_TEXTS = [
    ("rocket", "ordinary_word", 2),
    ("table", "ordinary_word", 1),
    ("window", "ordinary_word", 1),
    ("e2b1c4f0", "hex_fragment", None),
]


def load_legacy_modules():
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR))

    bootstrap_path = SCRIPTS_DIR / "_bootstrap.py"
    bootstrap_spec = importlib.util.spec_from_file_location("master_stimuli_bootstrap", bootstrap_path)
    bootstrap_module = importlib.util.module_from_spec(bootstrap_spec)
    sys.modules[bootstrap_spec.name] = bootstrap_module
    bootstrap_spec.loader.exec_module(bootstrap_module)

    byte_script_path = SCRIPTS_DIR / "06_byte_structure.py"
    byte_spec = importlib.util.spec_from_file_location("master_stimuli_byte_structure", byte_script_path)
    byte_module = importlib.util.module_from_spec(byte_spec)
    sys.modules[byte_spec.name] = byte_module
    byte_spec.loader.exec_module(byte_module)

    matched_script_path = SCRIPTS_DIR / "09_attention_theft.py"
    matched_spec = importlib.util.spec_from_file_location("master_stimuli_attention_theft", matched_script_path)
    matched_module = importlib.util.module_from_spec(matched_spec)
    sys.modules[matched_spec.name] = matched_module
    matched_spec.loader.exec_module(matched_module)

    return bootstrap_module, byte_module, matched_module


def parse_unicode_emoji_data(path):
    catalog = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "; fully-qualified" not in line:
            continue
        left_right = line.split(";", 1)
        code_text = left_right[0].strip()
        code_values = code_text.split()
        codepoints = []
        for value in code_values:
            codepoints.append(int(value, 16))
        description = line.split("#", 1)
        if len(description) < 2:
            continue
        name_parts = description[1].strip().split(maxsplit=2)
        if len(name_parts) < 3:
            continue
        name = name_parts[2]
        char = ""
        for codepoint in codepoints:
            char += chr(codepoint)
        key = tuple(codepoints)
        catalog[key] = {"char": char, "name": name, "codepoints": codepoints}
    return catalog


def codes_to_integers(code_strings):
    values = []
    for code_string in code_strings:
        values.append(int(code_string, 16))
    return values


def codes_to_char(code_strings):
    char = ""
    for codepoint in codes_to_integers(code_strings):
        char += chr(codepoint)
    return char


def text_codepoints(text):
    values = []
    for char in text:
        values.append(ord(char))
    return values


def format_codepoints(codepoints):
    formatted = []
    for codepoint in codepoints:
        formatted.append("U+" + format(codepoint, "04X"))
    return formatted


def format_hex_bytes(raw_bytes):
    formatted = []
    for value in raw_bytes:
        formatted.append(format(value, "02X"))
    return " ".join(formatted)


def get_token_record(tokenizer, token_id, token_bytes):
    raw_bytes = token_bytes(tokenizer, token_id)
    token_text = tokenizer.convert_ids_to_tokens(token_id)
    return {
        "token": token_text,
        "bytes": format_hex_bytes(raw_bytes),
        "lead_byte": byte_module.lead_byte_tag(raw_bytes),
    }


def audit_text(tokenizer, text, lead_byte_tag_function, token_bytes_function):
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoding["input_ids"])
    offsets = []
    for pair in encoding["offset_mapping"]:
        offsets.append((pair[0], pair[1]))

    try:
        token_indices = locate_span_by_offsets(offsets, 0, len(text))
    except Exception as error:
        return None, "span location failed: " + str(error)

    if len(token_indices) == 0:
        return None, "tokenizer returned an empty span"

    span_ids = []
    for token_index in token_indices:
        span_ids.append(input_ids[token_index])

    decoded = tokenizer.decode(
        span_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if decoded != text:
        return None, "decode round-trip mismatch: " + repr(decoded)

    tokens = []
    actual_leads = []
    for token_id in span_ids:
        raw = token_bytes_function(tokenizer, token_id)
        tokens.append(
            {
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "bytes": format_hex_bytes(raw),
                "lead_byte": lead_byte_tag_function(raw),
            }
        )
        actual_leads.append(lead_byte_tag_function(raw))

    return {
        "ids": span_ids,
        "tokens": tokens,
        "actual_fragment_count": len(span_ids),
        "actual_lead_bytes": actual_leads,
        "decoded": decoded,
    }, None


def expected_first_lead(codepoints, lead_byte_tag_function):
    if len(codepoints) == 0:
        return "other"
    first_char = chr(codepoints[0])
    return lead_byte_tag_function(first_char.encode("utf-8"))


def expected_f0_9f_count(codepoints):
    count = 0
    for codepoint in codepoints:
        encoded = chr(codepoint).encode("utf-8")
        if len(encoded) >= 2 and encoded[0] == 0xF0 and encoded[1] == 0x9F:
            count += 1
    return count


def expected_count_for_multicodepoint(codepoints):
    count = 0
    for codepoint in codepoints:
        if codepoint == 0x200D or codepoint == 0xFE0F:
            count += 2
        else:
            count += 3
    return count


def build_emoji_record(
    char,
    name,
    codepoints,
    expected_count,
    lead_byte_tag_function,
    token_bytes_function,
    extra_fields=None,
):
    audit, failure = audit_text(
        tokenizer,
        char,
        lead_byte_tag_function,
        token_bytes_function,
    )
    if failure is not None:
        return None, failure

    expected_lead = expected_first_lead(codepoints, lead_byte_tag_function)
    mismatch_flags = []
    first_actual_lead = audit["actual_lead_bytes"][0]
    if first_actual_lead != expected_lead:
        mismatch_flags.append(
            "expected first lead " + expected_lead + ", tokenizer gave " + first_actual_lead
        )
    if audit["actual_fragment_count"] != expected_count:
        mismatch_flags.append(
            "expected " + str(expected_count) + " fragments, tokenizer gave "
            + str(audit["actual_fragment_count"])
        )

    record = {
        "char": char,
        "emoji": char,
        "name": name,
        "codepoints": format_codepoints(codepoints),
        "expected_lead_byte": expected_lead,
        "expected_fragment_count": expected_count,
        "actual_fragment_count": audit["actual_fragment_count"],
        "actual_lead_bytes": audit["actual_lead_bytes"],
        "token_split": audit["tokens"],
        "mismatch_flags": mismatch_flags,
    }
    if extra_fields is not None:
        for key in extra_fields:
            record[key] = extra_fields[key]
    return record, None


def load_tokenizer_and_helpers():
    bootstrap_module, byte_module, matched_module = load_legacy_modules()
    model_config = bootstrap_module.load_config()
    model_entry = model_config["primary"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_entry["name"],
        revision=model_entry["revision"],
        use_fast=True,
        trust_remote_code=False,
        clean_up_tokenization_spaces=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The pinned Pythia tokenizer must be fast for offset verification")
    return tokenizer, model_entry, byte_module, matched_module


def build_a_group(catalog, lead_byte_tag_function, token_bytes_function, dropped, mismatches):
    items = []
    for block_label, range_start, range_end, candidate_codes in A_BLOCKS:
        block_count = 0
        for code_string in candidate_codes:
            codepoint = int(code_string, 16)
            if codepoint < range_start or codepoint > range_end:
                dropped.append({"group": "A_smp_pictographs", "item": code_string, "reason": "outside declared block"})
                continue
            catalog_entry = catalog.get((codepoint,))
            if catalog_entry is None:
                dropped.append({"group": "A_smp_pictographs", "item": code_string, "reason": "not a single fully-qualified Unicode 15.1 emoji"})
                continue
            char = catalog_entry["char"]
            name = catalog_entry["name"]
            extra = {"unicode_block": block_label}
            if code_string in A_GLOSSES:
                extra["gloss"] = A_GLOSSES[code_string]
            record, failure = build_emoji_record(
                char,
                name,
                [codepoint],
                3,
                lead_byte_tag_function,
                token_bytes_function,
                extra,
            )
            if failure is not None:
                dropped.append({"group": "A_smp_pictographs", "item": code_string, "reason": failure})
                continue
            record["control_id"] = "A-" + format(len(items) + 1, "03d")
            if len(record["mismatch_flags"]) > 0:
                mismatches.append({"group": "A_smp_pictographs", "char": char, "flags": record["mismatch_flags"]})
            items.append(record)
            block_count += 1
        if block_count != TARGET_A_PER_BLOCK:
            raise RuntimeError(
                block_label + " selected " + str(block_count) + " items; expected "
                + str(TARGET_A_PER_BLOCK)
            )
    return {
        "stratum": "A_smp_pictographs",
        "tests": "Does the leading F0 9F token attract attention across varied trailing bytes?",
        "lead_byte_token": "F0 9F",
        "predicted_sinks": 1,
        "predicted_sink_position": 0,
        "items": items,
    }


def sequence_catalog_entry(catalog, code_strings):
    codepoints = codes_to_integers(code_strings)
    key = tuple(codepoints)
    return catalog.get(key)


def build_b_group(catalog, tokenizer, byte_module, token_bytes_function, dropped, mismatches):
    all_items = []
    for range_index in range(len(B_RANGES)):
        range_label, range_start, range_end, target_count, candidates = B_RANGES[range_index]
        selected_in_range = 0
        for code_strings in candidates:
            base_codepoint = int(code_strings[0], 16)
            if base_codepoint < range_start or base_codepoint > range_end:
                dropped.append({"group": "B_bmp_e2_fragmenting", "item": " ".join(code_strings), "reason": "base codepoint outside declared range"})
                continue
            catalog_entry = sequence_catalog_entry(catalog, code_strings)
            if catalog_entry is None:
                dropped.append({"group": "B_bmp_e2_fragmenting", "item": " ".join(code_strings), "reason": "not a fully-qualified Unicode 15.1 sequence"})
                continue
            codepoints = catalog_entry["codepoints"]
            char = catalog_entry["char"]
            name = catalog_entry["name"]
            expected_count = 0
            for codepoint in codepoints:
                if codepoint == 0xFE0F:
                    expected_count += 2
                else:
                    expected_count += 3
            record, failure = build_emoji_record(
                char,
                name,
                codepoints,
                expected_count,
                byte_module.lead_byte_tag,
                token_bytes_function,
                {"unicode_block": range_label},
            )
            if failure is not None:
                dropped.append({"group": "B_bmp_e2_fragmenting", "item": char, "reason": failure})
                continue
            actual_first_lead = record["actual_lead_bytes"][0]
            if actual_first_lead != "E2":
                dropped.append({"group": "B_bmp_e2_fragmenting", "item": char, "reason": "tokenizer first lead was " + actual_first_lead + ", expected E2"})
                continue
            if record["actual_fragment_count"] < 2:
                dropped.append({"group": "B_bmp_e2_fragmenting", "item": char, "reason": "did not fragment into multiple tokens"})
                continue
            if len(record["mismatch_flags"]) > 0:
                mismatches.append({"group": "B_bmp_e2_fragmenting", "char": char, "flags": record["mismatch_flags"]})
            all_items.append(record)
            selected_in_range += 1
            if selected_in_range >= target_count:
                break
        if selected_in_range != target_count:
            raise RuntimeError(
                range_label + " produced " + str(selected_in_range) + " eligible items; expected "
                + str(target_count)
            )
    return {
        "stratum": "B_bmp_e2_fragmenting",
        "tests": "Does the F0 9F result differ from E2-leading BMP symbols that fragment?",
        "lead_byte_token": "E2",
        "predicted_sinks": 0,
        "items": all_items,
    }


def build_c_group(catalog, byte_module, token_bytes_function, dropped, mismatches):
    items = []
    for code_strings, name, expected_count in C_CANDIDATES:
        catalog_entry = sequence_catalog_entry(catalog, code_strings)
        if catalog_entry is None:
            dropped.append({"group": "C_multi_f0_9f", "item": " ".join(code_strings), "reason": "not a fully-qualified Unicode 15.1 sequence"})
            continue
        codepoints = catalog_entry["codepoints"]
        char = catalog_entry["char"]
        f0_count = expected_f0_9f_count(codepoints)
        record, failure = build_emoji_record(
            char,
            name,
            codepoints,
            expected_count,
            byte_module.lead_byte_tag,
            token_bytes_function,
            {"f0_9f_count": f0_count},
        )
        if failure is not None:
            dropped.append({"group": "C_multi_f0_9f", "item": char, "reason": failure})
            continue
        actual_f0_count = 0
        for lead in record["actual_lead_bytes"]:
            if lead == "F0 9F":
                actual_f0_count += 1
        if actual_f0_count != f0_count:
            record["mismatch_flags"].append(
                "expected " + str(f0_count) + " F0 9F lead tokens, tokenizer gave "
                + str(actual_f0_count)
            )
        if len(record["mismatch_flags"]) > 0:
            mismatches.append({"group": "C_multi_f0_9f", "char": char, "flags": record["mismatch_flags"]})
        items.append(record)
    if len(items) != TARGET_C:
        raise RuntimeError("C group selected " + str(len(items)) + " items; expected " + str(TARGET_C))
    return {
        "stratum": "C_multi_f0_9f",
        "tests": "Do multiple F0 9F-leading fragments align with multi-codepoint emoji components?",
        "lead_byte_token": "F0 9F (repeated)",
        "predicted_sinks": "= f0_9f_count",
        "items": items,
    }


def build_d_group(tokenizer, byte_module, token_bytes_function, dropped, mismatches):
    items = []
    accepted = 0
    for code_string in D_CANDIDATE_CODEPOINTS:
        codepoint = int(code_string, 16)
        char = chr(codepoint)
        name = unicodedata.name(char, "unassigned symbol")
        record, failure = build_emoji_record(
            char,
            name.lower(),
            [codepoint],
            1,
            byte_module.lead_byte_tag,
            token_bytes_function,
            {"unicode_block": "rare-whole-token candidate pool"},
        )
        if failure is not None:
            dropped.append({"group": "D_rare_whole_token", "item": code_string, "reason": failure})
            continue
        first_lead = record["actual_lead_bytes"][0]
        if first_lead != "E2":
            dropped.append({"group": "D_rare_whole_token", "item": code_string, "reason": "tokenizer first lead was " + first_lead + ", expected E2"})
            continue
        if record["actual_fragment_count"] != 1:
            dropped.append({"group": "D_rare_whole_token", "item": code_string, "reason": "not a single whole token; got " + str(record["actual_fragment_count"])})
            continue
        record["f0_9f_count"] = 0
        items.append(record)
        accepted += 1
        if accepted >= TARGET_D:
            break
    if len(items) != TARGET_D:
        raise RuntimeError("D group selected " + str(len(items)) + " items; expected " + str(TARGET_D))
    return {
        "stratum": "D_rare_whole_token",
        "tests": "Do rare E2-leading whole tokens also show a sink-like peak?",
        "lead_byte_token": "E2",
        "predicted_sinks": "unregistered exploratory alternative",
        "items": items,
    }


def build_reference_items(tokenizer, byte_module, token_bytes_function, dropped, mismatches):
    items = []
    for text, kind, expected_count in REFERENCE_TEXTS:
        codepoints = []
        for char in text:
            codepoints.append(ord(char))
        audit, failure = audit_text(
            tokenizer,
            text,
            byte_module.lead_byte_tag,
            token_bytes_function,
        )
        if failure is not None:
            dropped.append({"group": "R_reference", "item": text, "reason": failure})
            continue
        expected = expected_count
        if expected is None:
            expected = audit["actual_fragment_count"]
        expected_lead = "other"
        if len(audit["actual_lead_bytes"]) > 0:
            expected_lead = audit["actual_lead_bytes"][0]
        mismatch_flags = []
        if expected != audit["actual_fragment_count"]:
            mismatch_flags.append(
                "expected " + str(expected) + " fragments, tokenizer gave "
                + str(audit["actual_fragment_count"])
            )
            mismatches.append({"group": "R_reference", "char": text, "flags": mismatch_flags})
        items.append(
            {
                "text": text,
                "char": text,
                "name": kind.replace("_", " "),
                "kind": kind,
                "codepoints": format_codepoints(codepoints),
                "expected_lead_byte": expected_lead,
                "expected_fragment_count": expected,
                "actual_fragment_count": audit["actual_fragment_count"],
                "actual_lead_bytes": audit["actual_lead_bytes"],
                "token_split": audit["tokens"],
                "mismatch_flags": mismatch_flags,
            }
        )
    return items


def get_template_contexts(matched_module):
    templates = []
    specs = matched_module.TEMPLATE_SPECS
    for spec in specs:
        templates.append(matched_module.build_template(spec))
    return templates


def token_counts_across_templates(matched_module, tokenizer, text, templates):
    counts = []
    for template in templates:
        count = matched_module.insert_token_count(
            tokenizer,
            template["before"],
            text,
            template["after"],
        )
        counts.append(count)
    return counts


def find_hex_control(matched_module, tokenizer, target_count, templates):
    first = templates[0]
    helper_candidate = matched_module.hex_of_token_count(
        tokenizer,
        target_count,
        first["before"],
        first["after"],
        max_len=CONTROL_MAX_HEX_LENGTH,
    )
    if helper_candidate is not None:
        helper_counts = token_counts_across_templates(
            matched_module,
            tokenizer,
            helper_candidate,
            templates,
        )
        all_match = True
        for count in helper_counts:
            if count != target_count:
                all_match = False
        if all_match:
            return helper_candidate, helper_counts

    pattern = matched_module.HEX_PATTERN
    for length in range(1, CONTROL_MAX_HEX_LENGTH + 1):
        candidate = pattern[:length]
        counts = token_counts_across_templates(
            matched_module,
            tokenizer,
            candidate,
            templates,
        )
        all_match = True
        for count in counts:
            if count != target_count:
                all_match = False
        if all_match:
            return candidate, counts
    return None, []


def repeat_control_char(char, repeat_count):
    parts = []
    for index in range(repeat_count):
        parts.append(char)
    return " ".join(parts)


def find_rare_control(matched_module, tokenizer, target_count, templates, rare_items, offset):
    if len(rare_items) == 0:
        return None, None, []
    for step in range(len(rare_items)):
        rare_index = (offset + step) % len(rare_items)
        rare_item = rare_items[rare_index]
        rare_char = rare_item["char"]
        max_repeats = target_count + 2
        for repeat_count in range(1, max_repeats + 1):
            candidate = repeat_control_char(rare_char, repeat_count)
            counts = token_counts_across_templates(
                matched_module,
                tokenizer,
                candidate,
                templates,
            )
            all_match = True
            for count in counts:
                if count != target_count:
                    all_match = False
            if not all_match:
                continue
            audit, failure = audit_text(
                tokenizer,
                candidate,
                byte_module.lead_byte_tag,
                token_bytes,
            )
            if failure is None:
                return candidate, rare_item, counts
    return None, None, []


def make_control_record(kind, text, target_count, context_counts, tokenizer, byte_module):
    audit, failure = audit_text(tokenizer, text, byte_module.lead_byte_tag, token_bytes)
    if failure is not None:
        return None, failure
    mismatch_flags = []
    for count in context_counts:
        if count != target_count:
            mismatch_flags.append(
                "matched context count expected " + str(target_count) + ", observed " + str(count)
            )
    if audit["actual_fragment_count"] != target_count:
        mismatch_flags.append(
            "isolated token count expected " + str(target_count) + ", observed "
            + str(audit["actual_fragment_count"])
        )
    expected_lead = "other"
    if len(audit["actual_lead_bytes"]) > 0:
        expected_lead = audit["actual_lead_bytes"][0]
    record = {
        "text": text,
        "char": text,
        "kind": kind,
        "name": kind.replace("_", " "),
        "codepoints": format_codepoints(text_codepoints(text)),
        "expected_lead_byte": expected_lead,
        "expected_fragment_count": target_count,
        "actual_fragment_count": audit["actual_fragment_count"],
        "actual_lead_bytes": audit["actual_lead_bytes"],
        "token_split": audit["tokens"],
        "per_template_token_counts": context_counts,
        "mismatch_flags": mismatch_flags,
    }
    return record, None


def build_generated_controls(a_items, d_items, tokenizer, matched_module, byte_module, dropped, mismatches):
    templates = get_template_contexts(matched_module)
    generated = []
    for index in range(len(a_items)):
        item = a_items[index]
        emoji = item["char"]
        target_count = item["actual_fragment_count"]
        control_id = item["control_id"]

        hex_text, hex_counts = find_hex_control(
            matched_module,
            tokenizer,
            target_count,
            templates,
        )
        if hex_text is None:
            dropped.append({"group": "R_reference.generated_controls", "item": control_id + ":hex", "reason": "no hex string matched token count across all fragment-matched carriers"})
            raise RuntimeError("No shared hex match for " + control_id)
        hex_record, hex_failure = make_control_record(
            "matched_hex",
            hex_text,
            target_count,
            hex_counts,
            tokenizer,
            byte_module,
        )
        if hex_failure is not None:
            dropped.append({"group": "R_reference.generated_controls", "item": control_id + ":hex", "reason": hex_failure})
            raise RuntimeError("Hex control failed round-trip for " + control_id)
        if len(hex_record["mismatch_flags"]) > 0:
            mismatches.append(
                {
                    "group": "R_reference.generated_controls",
                    "char": hex_record["text"],
                    "flags": hex_record["mismatch_flags"],
                }
            )

        rare_text, rare_source, rare_counts = find_rare_control(
            matched_module,
            tokenizer,
            target_count,
            templates,
            d_items,
            index,
        )
        if rare_text is None:
            dropped.append({"group": "R_reference.generated_controls", "item": control_id + ":rare", "reason": "no rare whole-token sequence matched across all fragment-matched carriers"})
            raise RuntimeError("No shared rare-character match for " + control_id)
        rare_record, rare_failure = make_control_record(
            "matched_rare_char",
            rare_text,
            target_count,
            rare_counts,
            tokenizer,
            byte_module,
        )
        if rare_failure is not None:
            dropped.append({"group": "R_reference.generated_controls", "item": control_id + ":rare", "reason": rare_failure})
            raise RuntimeError("Rare-character control failed round-trip for " + control_id)
        if len(rare_record["mismatch_flags"]) > 0:
            mismatches.append(
                {
                    "group": "R_reference.generated_controls",
                    "char": rare_record["text"],
                    "flags": rare_record["mismatch_flags"],
                }
            )
        rare_record["source_rare_char"] = rare_source["char"]
        rare_record["source_rare_name"] = rare_source["name"]

        item["control_id"] = control_id
        generated.append(
            {
                "control_id": control_id,
                "for_char": emoji,
                "matched_token_count": target_count,
                "carrier_rule": "same filler token count in each of the three scripts/09_attention_theft.py carriers",
                "hex_fragment": hex_record,
                "rare_char": rare_record,
            }
        )
    return generated


def group_items(groups):
    all_items = []
    names = [
        "A_smp_pictographs",
        "B_bmp_e2_symbols",
        "C_multi_f0_9f",
        "D_nonemoji_e2",
    ]
    for group_name in names:
        group = groups[group_name]
        for item in group["items"]:
            all_items.append(item)
    return all_items


def compute_content_hash(groups):
    canonical = json.dumps(
        groups,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def print_summary(groups, dropped, mismatches, model_entry, content_hash):
    print("MASTER STIMULUS BUILD SUMMARY — REVIEW BEFORE FINAL USE")
    print("Version: master-stimuli-v1.0")
    print("Tokenizer: " + model_entry["name"])
    print("Tokenizer revision: " + model_entry["revision"])
    print("Content SHA-256: " + content_hash)
    print("Selection counts:")
    count = 0
    group_names = [
        "A_smp_pictographs",
        "B_bmp_e2_symbols",
        "C_multi_f0_9f",
        "D_nonemoji_e2",
    ]
    for group_name in group_names:
        group_count = len(groups[group_name]["items"])
        count += group_count
        print("  " + group_name + ": " + str(group_count))
    print("  Total stratified emoji/symbol stimuli: " + str(count))
    print("  R_reference baseline items: " + str(len(groups["R_reference"]["items"])))
    print("  Per-emoji generated control sets: " + str(len(groups["R_reference"]["generated_controls"])))

    print("Example real tokenizer splits:")
    shown = 0
    for group_name in group_names:
        for item in groups[group_name]["items"]:
            if shown >= 6:
                break
            token_descriptions = []
            for token in item["token_split"]:
                token_descriptions.append(token["bytes"])
            print(
                "  " + item["char"] + " (" + item["name"] + "): "
                + " | ".join(token_descriptions)
            )
            shown += 1
        if shown >= 6:
            break

    print("Flagged mismatches:")
    if len(mismatches) == 0:
        print("  none")
    else:
        for entry in mismatches:
            print("  " + entry["group"] + " " + entry["char"] + ": " + "; ".join(entry["flags"]))

    print("Dropped items:")
    if len(dropped) == 0:
        print("  none")
    else:
        for entry in dropped:
            print("  " + entry["group"] + " " + entry["item"] + ": " + entry["reason"])

    print("Review gate: inspect the generated config and this summary before using it in final runs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    global tokenizer
    global byte_module
    global locate_span_by_offsets
    global token_bytes
    sys.path.insert(0, str(SRC_DIR))
    from eas.span import locate_span_by_offsets
    from c2.tokaudit import token_bytes
    tokenizer, model_entry, byte_module, matched_module = load_tokenizer_and_helpers()
    catalog = parse_unicode_emoji_data(UNICODE_DATA)
    dropped = []
    mismatches = []

    a_group = build_a_group(catalog, byte_module.lead_byte_tag, token_bytes, dropped, mismatches)
    b_group = build_b_group(catalog, tokenizer, byte_module, token_bytes, dropped, mismatches)
    c_group = build_c_group(catalog, byte_module, token_bytes, dropped, mismatches)
    d_group = build_d_group(tokenizer, byte_module, token_bytes, dropped, mismatches)

    d_items = d_group["items"]
    matched_controls = build_generated_controls(
        a_group["items"],
        d_items,
        tokenizer,
        matched_module,
        byte_module,
        dropped,
        mismatches,
    )
    reference_items = build_reference_items(
        tokenizer,
        byte_module,
        token_bytes,
        dropped,
        mismatches,
    )

    groups = {
        "A_smp_pictographs": a_group,
        "B_bmp_e2_symbols": b_group,
        "C_multi_f0_9f": c_group,
        "D_nonemoji_e2": d_group,
        "R_reference": {
            "tests": "Fixed ordinary-word and hex baselines plus controls linked to every A item.",
            "note": "The RGI/rare control list is outside the 120-item stratified set.",
            "items": reference_items,
            "generated_controls": matched_controls,
        },
    }

    content_hash = compute_content_hash(groups)
    build_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = {
        "pre_registration": {
            "version_id": "master-stimuli-v1.0",
            "status": "pre_registered_frozen_reviewed",
            "build_timestamp_utc": build_timestamp,
            "content_sha256": content_hash,
            "unicode_data": "Emoji 15.1 fully-qualified entries from data/unicode/emoji-test-15.1.txt",
            "verification_model": model_entry["name"],
            "verification_revision": model_entry["revision"],
            "selection_rules": [
                "A has 12 single-codepoint entries in each of the five declared supplementary-plane ranges, in fixed candidate order.",
                "B screens fixed BMP-symbol candidates in three codepoint ranges; keep E2-leading entries with at least two actual Pythia tokens, taking the declared quota per range.",
                "C has eight skin-tone sequences, six ZWJ/family sequences, and six regional-indicator flags; expected F0 9F count follows supplementary-plane codepoints.",
                "D screens the fixed E2 symbol candidate pool and keeps the first 15 single-token, round-trip entries.",
                "A expected token counts are three; B uses the UTF-8 byte-split expectation with FE0F expected as two tokens; C uses three tokens per scalar except ZWJ and FE0F, which are expected as two; D expects one whole token.",
                "Each A stimulus receives deterministic hex and rare-character controls matched to its token count across all three fragment-matched carrier templates.",
                "The 15 glosses are a fixed subset of A and use one-word meanings.",
                "Any decode round-trip failure or unlocatable span is dropped; expected/actual lead or fragment-count differences are retained and flagged.",
            ],
        },
        "meta": {
            "reduction": "per_fragment_head_aware",
            "carrier": "structured_probe",
            "per_group_min": 8,
            "bin_by": "lead_byte_token",
            "note": "Master stimuli are tokenizer-audited; inspect the pre-registration summary and status before final runs.",
        },
        "groups": groups,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )
    output_path.write_text(output_text, encoding="utf-8")

    print_summary(groups, dropped, mismatches, model_entry, content_hash)
    print("Wrote review candidate: " + str(output_path))


if __name__ == "__main__":
    byte_module = None
    tokenizer = None
    main()
