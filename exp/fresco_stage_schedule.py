"""Epoch-level FReSCO curriculum used by the Subject 460 training run."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from exp.config_utils import set_dotted_value, validate_hf_image_config


SCHEDULE_NAME = "subject460_100ep"


@dataclass(frozen=True)
class FrescoStage:
    start: int
    end: int
    lf: float
    uv: float
    seam: float
    hf: float

    @property
    def label(self) -> str:
        return f"[{self.start},{self.end})"


SUBJECT460_100EP: Tuple[FrescoStage, ...] = (
    FrescoStage(0, 10, 0.25, 0.00, 0.00, 0.00),
    FrescoStage(10, 30, 0.15, 0.01, 0.00, 0.00),
    FrescoStage(30, 48, 0.10, 0.02, 0.00, 0.00),
    FrescoStage(48, 70, 0.05, 0.03, 0.05, 0.00),
    FrescoStage(70, 90, 0.02, 0.03, 0.02, 0.05),
    FrescoStage(90, 100, 0.01, 0.02, 0.01, 0.05),
)


def schedule_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("exp.fresco_schedule.enable", False))


def stage_for_epoch(epoch: int) -> FrescoStage:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError(f"epoch must be an integer, got {epoch!r}")
    for stage in SUBJECT460_100EP:
        if stage.start <= epoch < stage.end:
            return stage
    raise ValueError(f"Epoch {epoch} is outside the Subject 460 schedule [0,100)")


def paired_sampler_required(config: Dict[str, Any]) -> bool:
    return (
        bool(config.get("exp.lf_uv.enable", False))
        and bool(config.get("exp.lf_uv.loss.enable", False))
        and float(config.get("exp.lf_uv.loss.weight", 0.0)) > 0.0
    )


def apply_subject460_stage(config: Dict[str, Any], epoch: int) -> FrescoStage:
    """Apply one stage without touching model, optimizer, or base MeGA settings."""
    configured_name = config.get("exp.fresco_schedule.name", SCHEDULE_NAME)
    if configured_name != SCHEDULE_NAME:
        raise ValueError(
            f"Unsupported FReSCO schedule {configured_name!r}; expected {SCHEDULE_NAME!r}"
        )
    stage = stage_for_epoch(epoch)
    overrides = {
        "exp.lf_img.enable": stage.lf > 0.0,
        "exp.lf_img.lambda": stage.lf,
        # The umbrella remains enabled when either UV consumer needs baking.
        "exp.lf_uv.enable": stage.uv > 0.0 or stage.seam > 0.0,
        "exp.lf_uv.loss.enable": stage.uv > 0.0,
        "exp.lf_uv.loss.weight": stage.uv,
        "exp.lf_uv.seam.enable": stage.seam > 0.0,
        # Eq. (12) has no additional outer multiplier. The scheduled Seam
        # value is the pair coefficient; the configured TV coefficient stays fixed.
        "exp.lf_uv.seam.weight_pair": stage.seam,
        "exp.hf_img.enable": stage.hf > 0.0,
        "exp.hf_img.lambda": stage.hf,
    }
    for key, value in overrides.items():
        set_dotted_value(config, key, value)
    if stage.hf > 0.0:
        validate_hf_image_config(config, context=f"{SCHEDULE_NAME} epoch {epoch} HF-IMG")
    return stage


def effective_stage_dict(config: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    stage = apply_subject460_stage(config, epoch)
    return {
        "epoch": epoch,
        "interval": stage.label,
        "lf_enable": bool(config["exp.lf_img.enable"]),
        "lf_weight": float(config["exp.lf_img.lambda"]),
        "uv_enable": bool(config["exp.lf_uv.loss.enable"]),
        "uv_weight": float(config["exp.lf_uv.loss.weight"]),
        "seam_enable": bool(config["exp.lf_uv.seam.enable"]),
        "seam_pair_weight": float(config["exp.lf_uv.seam.weight_pair"]),
        "hf_enable": bool(config["exp.hf_img.enable"]),
        "hf_weight": float(config["exp.hf_img.lambda"]),
        "paired_sampler": paired_sampler_required(config),
    }
