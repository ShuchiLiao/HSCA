
from dataclasses import dataclass
from typing import Optional


SUPPORTED_ROUTE_NAMES = ("panns", "ast", "beats", "beats_adapt", "byola", "ead")


@dataclass(frozen=True)
class RouteConfig:
    """Centralized input protocol for one representation-learning route.

    This file only defines route-level configuration. It does not read data,
    run models, or perform any I/O.
    """

    route_name: str
    target_sr: int
    frontend_type: str          # "waveform" | "logmel" | "official_beats" | "engineered_waveform"
    n_mels: Optional[int] = None
    frame_ms: Optional[float] = None
    hop_ms: Optional[float] = None
    fmin: Optional[float] = None
    fmax: Optional[float] = None
    norm_mode: str = "none"    # "none" | "global_stats" | "official"
    default_stats_path: Optional[str] = None
    default_checkpoint_path: Optional[str] = None
    batch_input_layout: str = "B_L"  # "B_L" | "B_T_F" | "B_1_F_T"


def get_route_config(route_name: str) -> RouteConfig:
    """Return the complete configuration for one fixed route.

    Parameters
    ----------
    route_name:
        One of: "panns", "ast", "beats", "beats_adapt", "byola", "ead".

    Returns
    -------
    RouteConfig
        The full input protocol for the requested route.

    Raises
    ------
    ValueError
        If ``route_name`` is not one of the supported fixed route names.
    """
    name = str(route_name).strip().lower()

    if name == "panns":
        return RouteConfig(
            route_name="panns",
            target_sr=32000,
            frontend_type="waveform",
            n_mels=None,
            frame_ms=None,
            hop_ms=None,
            fmin=None,
            fmax=None,
            norm_mode="none",
            default_stats_path=None,
            default_checkpoint_path=None,
            batch_input_layout="B_L",
        )

    if name == "ast":
        return RouteConfig(
            route_name="ast",
            target_sr=16000,
            frontend_type="logmel",
            n_mels=128,
            frame_ms=25.0,
            hop_ms=10.0,
            fmin=50.0,
            fmax=8000.0,
            norm_mode="global_stats",
            default_stats_path="artifacts/stats/ast_stats.json",
            default_checkpoint_path=None,
            batch_input_layout="B_T_F",
        )

    if name == "beats":
        return RouteConfig(
            route_name="beats",
            target_sr=16000,
            frontend_type="official_beats",
            n_mels=None,
            frame_ms=None,
            hop_ms=None,
            fmin=None,
            fmax=None,
            norm_mode="official",
            default_stats_path=None,
            default_checkpoint_path=None,
            batch_input_layout="B_L",
        )

    if name == "beats_adapt":
        return RouteConfig(
            route_name="beats_adapt",
            target_sr=16000,
            frontend_type="official_beats",
            n_mels=None,
            frame_ms=None,
            hop_ms=None,
            fmin=None,
            fmax=None,
            norm_mode="official",
            default_stats_path=None,
            default_checkpoint_path="artifacts/checkpoints/beats_adapt/best.pt",
            batch_input_layout="B_L",
        )

    if name == "byola":
        return RouteConfig(
            route_name="byola",
            target_sr=16000,
            frontend_type="logmel",
            n_mels=64,
            frame_ms=25.0,
            hop_ms=10.0,
            fmin=50.0,
            fmax=8000.0,
            norm_mode="global_stats",
            default_stats_path="artifacts/stats/byola_stats.json",
            default_checkpoint_path="artifacts/checkpoints/byola/best.pt",
            batch_input_layout="B_1_F_T",
        )

    if name == "ead":
        return RouteConfig(
            route_name="ead",
            target_sr=8000,
            frontend_type="engineered_waveform",
            n_mels=None,
            frame_ms=None,
            hop_ms=None,
            fmin=None,
            fmax=None,
            norm_mode="none",
            default_stats_path=None,
            default_checkpoint_path=None,
            batch_input_layout="B_L",
        )

    raise ValueError(
        f"Unsupported route_name={route_name!r}. "
        f"Expected one of: {', '.join(SUPPORTED_ROUTE_NAMES)}"
    )
