"""
Config Editor page.
Load, edit, validate and save the training YAML config. Hot-reload notification.
No async/await.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "config",
    "training_config.yaml",
)

# Top-level keys that must be present in a valid training config.
REQUIRED_TOP_LEVEL_KEYS = {"env", "training", "risk"}

# Keys that trigger a warning but are not errors (recommended but optional).
RECOMMENDED_KEYS = {"ensemble", "validation", "hyperopt", "regime"}

# ── Pure helper functions ─────────────────────────────────────────────────────


def load_config_yaml(config_path: str) -> Tuple[str, Dict[str, Any]]:
    """Load config YAML from *config_path*.

    Returns:
        (yaml_string, parsed_dict)  — both empty on any error.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s", config_path)
        return "", {}

    try:
        raw = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw) or {}
        return raw, parsed
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Could not load config %s: %s", config_path, exc)
        return "", {}


def validate_config_yaml(yaml_str: str) -> Tuple[bool, List[str]]:
    """Validate a YAML string for use as a training config.

    Checks:
    1. Valid YAML syntax
    2. All REQUIRED_TOP_LEVEL_KEYS present
    3. Warnings for missing RECOMMENDED_KEYS

    Returns:
        (is_valid: bool, messages: List[str])
        Messages contain error strings (prefixed "ERROR:") and warnings ("WARNING:").
        is_valid is True only when there are no ERROR messages.
    """
    messages: List[str] = []

    if not yaml_str.strip():
        messages.append("ERROR: Config is empty.")
        return False, messages

    try:
        parsed = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        messages.append(f"ERROR: Invalid YAML syntax — {exc}")
        return False, messages

    if not isinstance(parsed, dict):
        messages.append("ERROR: Config must be a YAML mapping (dict) at the top level.")
        return False, messages

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in parsed:
            messages.append(f"ERROR: Missing required top-level key: '{key}'")

    for key in RECOMMENDED_KEYS:
        if key not in parsed:
            messages.append(f"WARNING: Recommended key '{key}' not present.")

    is_valid = not any(m.startswith("ERROR:") for m in messages)
    return is_valid, messages


def save_config_yaml(config_path: str, yaml_str: str) -> bool:
    """Validate then save *yaml_str* to *config_path*.

    Returns True on success, False on validation failure or I/O error.
    """
    is_valid, messages = validate_config_yaml(yaml_str)
    if not is_valid:
        logger.warning("Config not saved due to validation errors: %s", messages)
        return False

    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml_str, encoding="utf-8")
        logger.info("Config saved to %s", config_path)
        return True
    except OSError as exc:
        logger.warning("Could not save config to %s: %s", config_path, exc)
        return False


def get_config_schema() -> Dict[str, Any]:
    """Return a minimal schema description of expected config structure.

    Used for documentation / tooltip purposes in the UI.
    """
    return {
        "required": sorted(REQUIRED_TOP_LEVEL_KEYS),
        "recommended": sorted(RECOMMENDED_KEYS),
        "description": (
            "Training config for the multi-agent RL trading bot. "
            "Must contain 'env', 'training', and 'risk' sections."
        ),
    }


def diff_configs(
    original: Dict[str, Any], updated: Dict[str, Any], prefix: str = ""
) -> List[str]:
    """Return a list of human-readable diff lines between two config dicts."""
    changes: List[str] = []
    all_keys = set(original) | set(updated)

    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in original:
            changes.append(f"ADDED   {full_key}: {updated[key]!r}")
        elif key not in updated:
            changes.append(f"REMOVED {full_key}: {original[key]!r}")
        elif isinstance(original[key], dict) and isinstance(updated[key], dict):
            changes.extend(diff_configs(original[key], updated[key], prefix=full_key))
        elif original[key] != updated[key]:
            changes.append(f"CHANGED {full_key}: {original[key]!r} → {updated[key]!r}")

    return changes


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_config_editor() -> None:
    """Render the Config Editor Streamlit page (synchronous)."""
    import streamlit as st  # local import so module is mockable in tests

    st.title("Config Editor")

    # ── Config path selection ─────────────────────────────────────────────
    config_path = st.sidebar.text_input(
        "Config file path",
        value=st.session_state.get("config_editor_path", DEFAULT_CONFIG_PATH),
    )
    st.session_state["config_editor_path"] = config_path

    schema = get_config_schema()
    with st.sidebar.expander("Required keys"):
        for k in schema["required"]:
            st.sidebar.markdown(f"- `{k}`")
    with st.sidebar.expander("Recommended keys"):
        for k in schema["recommended"]:
            st.sidebar.markdown(f"- `{k}`")

    # ── Load ──────────────────────────────────────────────────────────────
    if "config_editor_text" not in st.session_state:
        raw, _ = load_config_yaml(config_path)
        st.session_state["config_editor_text"] = raw
        st.session_state["config_editor_original"] = raw

    col_reload, col_save, col_download = st.columns(3)

    if col_reload.button("Reload from disk"):
        raw, _ = load_config_yaml(config_path)
        if raw:
            st.session_state["config_editor_text"] = raw
            st.session_state["config_editor_original"] = raw
            st.success("Reloaded from disk.")
        else:
            st.error(f"Could not load: {config_path}")

    # ── Editor ────────────────────────────────────────────────────────────
    edited_yaml = st.text_area(
        "Edit config (YAML)",
        value=st.session_state.get("config_editor_text", ""),
        height=500,
        key="config_yaml_editor",
    )
    st.session_state["config_editor_text"] = edited_yaml

    # ── Validation panel ──────────────────────────────────────────────────
    is_valid, messages = validate_config_yaml(edited_yaml)

    with st.expander("Validation", expanded=not is_valid):
        if is_valid:
            st.success("Config is valid.")
        for msg in messages:
            if msg.startswith("ERROR:"):
                st.error(msg)
            else:
                st.warning(msg)

    # ── Diff panel ────────────────────────────────────────────────────────
    original_raw = st.session_state.get("config_editor_original", "")
    if original_raw and original_raw != edited_yaml:
        try:
            original_dict = yaml.safe_load(original_raw) or {}
            edited_dict = yaml.safe_load(edited_yaml) or {}
            diffs = diff_configs(original_dict, edited_dict)
            if diffs:
                with st.expander(f"Changes ({len(diffs)})"):
                    for d in diffs:
                        st.code(d)
        except yaml.YAMLError:
            pass  # don't show diff if yaml is broken

    # ── Save ─────────────────────────────────────────────────────────────
    if col_save.button("Save", disabled=not is_valid, type="primary"):
        if save_config_yaml(config_path, edited_yaml):
            st.session_state["config_editor_original"] = edited_yaml
            st.success(f"Saved to `{config_path}`. Restart training to apply changes.")
        else:
            st.error("Failed to save config. Check validation errors and file permissions.")

    # ── Download ─────────────────────────────────────────────────────────
    col_download.download_button(
        label="Download",
        data=edited_yaml.encode("utf-8"),
        file_name=Path(config_path).name,
        mime="text/yaml",
    )
