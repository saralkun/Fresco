"""Canonical dotted/nested configuration handling for FReSCO losses."""

import warnings


HF_IMAGE_REQUIRED_KEYS = (
    "exp.hf_img.kernel1", "exp.hf_img.sigma1",
    "exp.hf_img.kernel2", "exp.hf_img.sigma2",
    "exp.hf_img.lambda_h1", "exp.hf_img.lambda_h2",
)


def _flatten(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _set_nested(mapping, dotted_key, value):
    parts = dotted_key.split(".")
    node = mapping
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def set_dotted_value(config, dotted_key, value):
    """Set both aliases of one canonical dotted configuration value."""
    config[dotted_key] = value
    _set_nested(config, dotted_key, value)


def normalize_dotted_config(config, explicit_overrides=None):
    """Make dotted and nested aliases agree; explicit/flat values win.

    Shell ``extra_config`` keys are dotted, so their precedence is retained.
    Conflicts in the YAML are surfaced instead of silently using defaults.
    """
    explicit_overrides = explicit_overrides or {}
    nested_flat = _flatten({k: v for k, v in config.items() if isinstance(v, dict)})
    flat_values = {k: v for k, v in config.items() if "." in str(k)}
    keys = set(nested_flat) | set(flat_values) | set(explicit_overrides)
    for key in sorted(keys):
        nested = nested_flat.get(key)
        flat = flat_values.get(key)
        if key in explicit_overrides:
            value = explicit_overrides[key]
        elif key in flat_values:
            value = flat
        else:
            value = nested
        if key in flat_values and key in nested_flat and flat != nested:
            warnings.warn(
                f"Configuration conflict for {key!r}: nested={nested!r}, flat={flat!r}; "
                f"using {value!r}", RuntimeWarning
            )
        config[key] = value
        _set_nested(config, key, value)
    return config


def validate_hf_image_config(config, context="HF-IMG"):
    """Return validated paper HF parameters without inventing missing values."""
    missing = [key for key in HF_IMAGE_REQUIRED_KEYS if key not in config or config[key] is None]
    if missing:
        raise ValueError(f"{context} missing required configuration keys: {', '.join(missing)}")
    values = {}
    for dotted_key in HF_IMAGE_REQUIRED_KEYS:
        value = config[dotted_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{context} configuration {dotted_key} must be numeric, got {value!r}"
            )
        values[dotted_key.rsplit(".", 1)[-1]] = value

    for key in ("kernel1", "kernel2"):
        kernel = int(values[key])
        if kernel <= 0 or kernel % 2 == 0 or kernel != values[key]:
            raise ValueError(f"{context} {key} must be a positive odd integer, got {values[key]!r}")
        values[key] = kernel
    for key in ("sigma1", "sigma2"):
        sigma = float(values[key])
        if sigma <= 0:
            raise ValueError(f"{context} {key} must be positive, got {values[key]!r}")
        values[key] = sigma
    if values["sigma1"] >= values["sigma2"]:
        raise ValueError(
            f"{context} requires sigma1 < sigma2, got {values['sigma1']} >= {values['sigma2']}"
        )
    for key in ("lambda_h1", "lambda_h2"):
        weight = float(values[key])
        if weight < 0:
            raise ValueError(f"{context} {key} must be non-negative, got {values[key]!r}")
        values[key] = weight
    return values


def effective_fresco_config(config):
    keys = (
        "exp.lf_img.enable", "exp.lf_img.lambda", "exp.lf_img.kernel", "exp.lf_img.sigma",
        "exp.hf_img.enable", "exp.hf_img.lambda", "exp.hf_img.kernel1", "exp.hf_img.sigma1",
        "exp.hf_img.kernel2", "exp.hf_img.sigma2", "exp.hf_img.lambda_h1", "exp.hf_img.lambda_h2",
        "exp.lf_uv.enable", "exp.lf_uv.every", "exp.lf_uv.uv_res", "exp.lf_uv.loss.enable",
        "exp.lf_uv.loss.weight", "exp.lf_uv.seam.enable", "exp.lf_uv.seam.weight_pair",
        "exp.lf_uv.seam.weight_tv", "exp.lf_uv.seam.band_px",
    )
    return {key: config.get(key) for key in keys}
