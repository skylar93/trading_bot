"""
Config secret resolver.

Walks a config dict tree and replaces every ``secret_ref: "<KEY>"`` leaf
with the resolved secret value from the given provider.

The resolved dict is a *copy* — the original is not modified.

Example YAML input::

    paper_trading:
      api_key_ref: "EXCHANGE_BINANCE_KEY"
      api_secret_ref: "EXCHANGE_BINANCE_SECRET"

After resolution, the dict contains::

    paper_trading:
      api_key: "<actual-key>"
      api_secret: "<actual-secret>"

Convention: any key ending in ``_ref`` whose value is a non-empty string
is treated as a secret reference.  The ``_ref`` suffix is stripped and the
resolved value is placed under the resulting key name.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from .secret_provider import SecretProvider, get_default_provider

logger = logging.getLogger(__name__)

_REF_SUFFIX = "_ref"


def resolve_secrets(
    config: dict[str, Any],
    provider: SecretProvider | None = None,
) -> dict[str, Any]:
    """Return a deep copy of *config* with all ``*_ref`` keys resolved.

    Parameters
    ----------
    config:
        Parsed YAML/dict config (may be nested).
    provider:
        SecretProvider to use.  Defaults to ``get_default_provider()``.

    Raises
    ------
    KeyError
        If a referenced secret key is not found in the provider.
    """
    if provider is None:
        provider = get_default_provider()
    return _resolve_node(copy.deepcopy(config), provider)


def _resolve_node(node: Any, provider: SecretProvider) -> Any:
    if isinstance(node, dict):
        resolved: dict[str, Any] = {}
        for k, v in node.items():
            if isinstance(k, str) and k.endswith(_REF_SUFFIX) and isinstance(v, str) and v:
                # e.g. api_key_ref: "EXCHANGE_BINANCE_KEY"  →  api_key: "<value>"
                bare_key = k[: -len(_REF_SUFFIX)]
                secret_value = provider.get(v)
                resolved[bare_key] = secret_value
                logger.debug("Resolved secret_ref '%s' → key '%s'", v, bare_key)
            else:
                resolved[k] = _resolve_node(v, provider)
        return resolved
    if isinstance(node, list):
        return [_resolve_node(item, provider) for item in node]
    return node
