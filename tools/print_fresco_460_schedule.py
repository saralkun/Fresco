#!/usr/bin/env python3
"""Print and validate the Subject 460 FReSCO curriculum without importing torch."""

import json

from exp.fresco_stage_schedule import SUBJECT460_100EP, effective_stage_dict


def main():
    config = {
        "exp.fresco_schedule.name": "subject460_100ep",
        "exp.hf_img.kernel1": 3,
        "exp.hf_img.sigma1": 0.8,
        "exp.hf_img.kernel2": 5,
        "exp.hf_img.sigma2": 1.2,
        "exp.hf_img.lambda_h1": 1.0,
        "exp.hf_img.lambda_h2": 1.0,
    }
    for stage in SUBJECT460_100EP:
        print(json.dumps(effective_stage_dict(config, stage.start), sort_keys=True))


if __name__ == "__main__":
    main()
