#!/usr/bin/env python3
"""Shared Stage 2 role-aware activity taxonomy helpers."""

from __future__ import annotations

from typing import Iterable


ROLE_LABELS = {
    "sewing": ["idle", "get", "put", "align", "pass", "inspect", "sew", "adjust_machine", "uncertain"],
    "sew_support": ["idle", "get", "put", "align", "pass", "inspect", "sew", "adjust_machine", "uncertain"],
    "prep_pass": ["idle", "get", "put", "align", "pass", "inspect", "uncertain"],
    "generic": ["idle", "get", "put", "align", "pass", "inspect", "sew", "adjust_machine", "uncertain"],
}

ROLE_DESCRIPTIONS = {
    "sewing": "Sewing-capable station where the worker may get, place, align, inspect, pass, feed material to the machine, sew, or adjust the machine.",
    "sew_support": "Sewing-support station where align and fabric handling dominate, but machine engagement and short sewing bursts may still appear.",
    "prep_pass": "Preparation or pass-forward station where the worker mainly gets, aligns, inspects, places, and passes fabric onward instead of sewing.",
    "generic": "Unknown station role. Use the most conservative action label based on visible hand and fabric interaction.",
}

GEMINI_PROTOCOL_VERSION = "altersense_stage2_gemini_v1"

ROLE_CYCLE_PHASES = {
    "sewing": ["idle_phase", "align_phase", "place_phase", "sew_phase", "pickup_phase", "uncertain_phase"],
    "sew_support": ["idle_phase", "align_phase", "place_phase", "sew_phase", "pickup_phase", "uncertain_phase"],
    "prep_pass": ["idle_phase", "align_phase", "handoff_phase", "pickup_phase", "uncertain_phase"],
    "generic": ["idle_phase", "align_phase", "place_phase", "sew_phase", "pickup_phase", "handoff_phase", "uncertain_phase"],
}

PHASE_CYCLE_STEPS_BY_ROLE = {
    "sewing": [["align_phase"], ["place_phase"], ["sew_phase"], ["pickup_phase"]],
    "sew_support": [["align_phase"], ["place_phase"], ["sew_phase"], ["pickup_phase"]],
    "prep_pass": [["align_phase"], ["handoff_phase"], ["pickup_phase"]],
    "generic": [["align_phase"], ["place_phase", "handoff_phase"], ["sew_phase", "pickup_phase"]],
}

PHASE_KPI_LABELS_BY_ROLE = {
    "sewing": {
        "productive": {"place_phase", "sew_phase", "pickup_phase"},
        "support": {"align_phase"},
        "npt": {"idle_phase", "uncertain_phase"},
    },
    "sew_support": {
        "productive": {"align_phase", "place_phase", "sew_phase", "pickup_phase"},
        "support": set(),
        "npt": {"idle_phase", "uncertain_phase"},
    },
    "prep_pass": {
        "productive": {"handoff_phase", "pickup_phase"},
        "support": {"align_phase"},
        "npt": {"idle_phase", "uncertain_phase"},
    },
    "generic": {
        "productive": {"place_phase", "handoff_phase", "sew_phase", "pickup_phase"},
        "support": {"align_phase"},
        "npt": {"idle_phase", "uncertain_phase"},
    },
}

LABEL_TO_PHASE_BY_ROLE = {
    "sewing": {
        "idle": "idle_phase",
        "uncertain": "uncertain_phase",
        "align": "align_phase",
        "inspect": "align_phase",
        "put": "place_phase",
        "pass": "place_phase",
        "sew": "sew_phase",
        "adjust_machine": "sew_phase",
        "get": "pickup_phase",
    },
    "sew_support": {
        "idle": "idle_phase",
        "uncertain": "uncertain_phase",
        "align": "align_phase",
        "inspect": "align_phase",
        "put": "place_phase",
        "pass": "place_phase",
        "sew": "sew_phase",
        "adjust_machine": "sew_phase",
        "get": "pickup_phase",
    },
    "prep_pass": {
        "idle": "idle_phase",
        "uncertain": "uncertain_phase",
        "align": "align_phase",
        "inspect": "align_phase",
        "put": "handoff_phase",
        "pass": "handoff_phase",
        "get": "pickup_phase",
    },
    "generic": {
        "idle": "idle_phase",
        "uncertain": "uncertain_phase",
        "align": "align_phase",
        "inspect": "align_phase",
        "put": "place_phase",
        "pass": "handoff_phase",
        "sew": "sew_phase",
        "adjust_machine": "sew_phase",
        "get": "pickup_phase",
    },
}

GEMINI_EXTRA_FIELDS = [
    "gemini_protocol_version",
    "gemini_cycle_phase",
    "gemini_motion_direction",
    "gemini_machine_engaged",
    "gemini_hands_on_material",
    "gemini_transition_ok",
    "gemini_safe_label",
    "gemini_schema_error",
    "gemini_json",
]

# Current default assumptions from repo findings. These can be overridden per-row in CSV.
STATION_ROLE_MAP = {
    "2": "sewing",
    "4": "sew_support",
}

KEYBOARD_HINTS = {
    "1": "idle",
    "2": "get",
    "3": "put",
    "4": "align",
    "5": "sew",
    "6": "pass",
    "7": "inspect",
    "8": "adjust_machine",
    "9": "uncertain",
}


def infer_station_role(row: dict) -> str:
    explicit = (row.get("station_role") or "").strip().lower()
    if explicit in ROLE_LABELS:
        return explicit
    station_id = str(row.get("station_id", "")).strip()
    return STATION_ROLE_MAP.get(station_id, "generic")


def allowed_labels_for_row(row: dict) -> list[str]:
    return ROLE_LABELS[infer_station_role(row)]


def role_description_for_row(row: dict) -> str:
    return ROLE_DESCRIPTIONS[infer_station_role(row)]


def normalize_label_for_row(label: str, row: dict) -> str:
    clean = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    allowed = set(allowed_labels_for_row(row))
    alias_map = {
        "place": "put",
        "place_fabric": "put",
        "align_fabric": "align",
        "prepare": "align",
        "prepare_fabric": "align",
        "handoff": "pass",
        "pass_bundle": "pass",
        "check": "inspect",
        "inspection": "inspect",
        "machine_adjustment": "adjust_machine",
    }
    clean = alias_map.get(clean, clean)
    return clean if clean in allowed else "uncertain"


def allowed_cycle_phases_for_row(row: dict) -> list[str]:
    return ROLE_CYCLE_PHASES[infer_station_role(row)]


def default_cycle_phase_for_label(label: str, row: dict) -> str:
    role = infer_station_role(row)
    mapping = LABEL_TO_PHASE_BY_ROLE.get(role, LABEL_TO_PHASE_BY_ROLE["generic"])
    return mapping.get(label, "uncertain_phase")


def normalize_cycle_phase_for_row(phase: str, row: dict, label: str = "") -> str:
    clean = (phase or "").strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "idle": "idle_phase",
        "align": "align_phase",
        "place": "place_phase",
        "put": "place_phase",
        "pass": "handoff_phase",
        "sew": "sew_phase",
        "adjust": "sew_phase",
        "get": "pickup_phase",
        "pickup": "pickup_phase",
        "uncertain": "uncertain_phase",
    }
    clean = alias_map.get(clean, clean)
    allowed = set(allowed_cycle_phases_for_row(row))
    if clean in allowed:
        return clean
    if label:
        fallback = default_cycle_phase_for_label(label, row)
        if fallback in allowed:
            return fallback
    return "uncertain_phase"


def phase_matches_label(label: str, cycle_phase: str, row: dict) -> bool:
    if not label:
        return False
    expected = default_cycle_phase_for_label(label, row)
    if label == "uncertain":
        return True
    return expected == cycle_phase


def keyboard_hints_for_row(row: dict) -> list[tuple[str, str]]:
    allowed = set(allowed_labels_for_row(row))
    return [(key, label) for key, label in KEYBOARD_HINTS.items() if label in allowed]


def accepted_statuses() -> set[str]:
    return {"done", "reviewed", "approved"}
