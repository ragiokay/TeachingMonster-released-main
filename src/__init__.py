from .config_schema import AppConfig
from .cursor.cursor import CursorModule
from .gemini_client import GeminiClient
from .outline.t2v_outline import T2VOutlineModule
from .outline.wrapper import Wrapper_3B1B, Wrapper_PPT
from .slides_ppt.slides_ppt import SlidesModule_PPT
from .tts.tts import TTSModule


def _missing_optional_class(class_name: str, import_path: str, err: Exception):
    class _MissingOptional:  # pragma: no cover - simple runtime guard
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"Optional module '{class_name}' is unavailable. "
                f"Expected import path: '{import_path}'. Original error: {err}"
            )

    _MissingOptional.__name__ = class_name
    return _MissingOptional


try:
    from .clarification.clarification import ClarificationModule
except Exception as _clarification_err:
    ClarificationModule = _missing_optional_class(
        "ClarificationModule",
        "src.clarification.clarification",
        _clarification_err,
    )

try:
    from .outline.v2v_outline import V2VOutlineModule
except Exception as _v2v_err:
    V2VOutlineModule = _missing_optional_class(
        "V2VOutlineModule", "src.outline.v2v_outline", _v2v_err
    )

try:
    from .slides_3B1B.slides_3B1B import SlidesModule_3B1B
except Exception as _slides_3b1b_err:
    SlidesModule_3B1B = _missing_optional_class(
        "SlidesModule_3B1B", "src.slides_3B1B.slides_3B1B", _slides_3b1b_err
    )

__all__ = [
    "AppConfig",
    "ClarificationModule",
    "CursorModule",
    "GeminiClient",
    "SlidesModule_3B1B",
    "SlidesModule_PPT",
    "T2VOutlineModule",
    "TTSModule",
    "V2VOutlineModule",
    "Wrapper_3B1B",
    "Wrapper_PPT",
]
