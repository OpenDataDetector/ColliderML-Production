"""Whether the digitisation stage writes vertex-to-track links.

The links come from VertexTrackLinkWriter, a Python module (``colliderml_vertex_links``)
that only the patched sim image ships (image/vertex_track_links). The config key
``vertexing_track_links`` takes:

  true    write the links; the stage fails if the module is missing (production)
  false   never write them
  auto    write them if the module can be imported, otherwise run without
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)

LINK_MODULE = "colliderml_vertex_links"


def link_writer_available() -> bool:
    return importlib.util.find_spec(LINK_MODULE) is not None


def want_vertex_track_links(setting, available: bool | None = None) -> bool:
    """Resolve the ``vertexing_track_links`` config value to a bool.

    ``available`` overrides the import check (tests)."""
    if isinstance(setting, str):
        value = setting.strip().lower()
        if value == "auto":
            if available is None:
                available = link_writer_available()
            if not available:
                logger.warning(
                    "vertexing_track_links: auto, and this image has no %s module: "
                    "vertices are written without track links", LINK_MODULE)
            return bool(available)
        if value in ("true", "false"):
            return value == "true"
        raise ValueError(f"vertexing_track_links must be true, false or auto, not {setting!r}")
    return bool(setting)
