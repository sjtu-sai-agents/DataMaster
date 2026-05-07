"""Feishu SDK Client factory

Create and cache lark_oapi.Client instances.
"""

from __future__ import annotations

import logging
from typing import Dict

import lark_oapi as lark

logger = logging.getLogger(__name__)

# Cache: key = (app_id, domain) -> Client
_client_cache: Dict[str, lark.Client] = {}


def create_feishu_client(
    app_id: str,
    app_secret: str,
    domain: str = "https://open.feishu.cn",
) -> lark.Client:
    """Create or retrieve a cached Feishu Client.

    Args:
        app_id: Feishu application App ID.
        app_secret: Feishu application App Secret.
        domain: Feishu API domain.

    Returns:
        A lark_oapi.Client instance.
    """
    cache_key = f"{app_id}@{domain}"

    if cache_key in _client_cache:
        logger.debug("Reusing cached Feishu client: %s", cache_key)
        return _client_cache[cache_key]

    # Domain -> lark domain constant mapping
    domain_map = {
        "https://open.feishu.cn": lark.FEISHU_DOMAIN,
        "https://open.larksuite.com": lark.LARK_DOMAIN,
    }
    lark_domain = domain_map.get(domain, domain)

    client = (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .domain(lark_domain)
        .log_level(lark.LogLevel.WARNING)
        .build()
    )

    _client_cache[cache_key] = client
    logger.info("Created Feishu client: %s", cache_key)
    return client
