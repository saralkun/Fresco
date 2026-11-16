# FReSCO

This release contains the FReSCO training and evaluation implementation used
for Subject 460. It is built on
[MeGA](https://github.com/conallwang/MeGA) and expects the preprocessed
NeRSemble data released for MeGA, together with the required FLAME assets and a
pretrained canonical hair checkpoint.

## Environment and data

Use the MeGA-compatible `mega` Conda environment. Dataset, FLAME, initial-point,
and pretrained-hair paths are read from the selected YAML configuration. The
Subject 460 configuration used by the server run is:

```text
configs/nersemble/460/full.yaml
```

Update its machine-specific paths before running on a new server. Local data,
checkpoints, caches, logs, and evaluation outputs are excluded by `.gitignore`.

## Training

Subject 460 is launched by the server training entry point:

```bash
PROJECT_DIR=/path/to/fresco_release \
OUTPUT_ROOT=/path/to/checkpoints/fresco_460_100ep \
RUN_NAME=subject460_100ep \
GPU_ID=0 \
bash scripts/train_fresco_460_100ep.sh --run
```

The same script supports command inspection, background execution, and strict
resume without changing its training logic:

```bash
PROJECT_DIR=/path/to/fresco_release bash scripts/train_fresco_460_100ep.sh --dry-run
PROJECT_DIR=/path/to/fresco_release SESSION_NAME=fresco460 bash scripts/train_fresco_460_100ep.sh --screen
PROJECT_DIR=/path/to/fresco_release RESUME=/path/to/checkpoint_latest.pth bash scripts/train_fresco_460_100ep.sh --run
```

The release implementation uses this fixed six-stage, 100-epoch curriculum:

| Epochs | LF-IMG | Cross-view UV | Seam pair | HF-IMG |
| --- | ---: | ---: | ---: | ---: |
| `[0,10)` | 0.25 | 0 | 0 | 0 |
| `[10,30)` | 0.15 | 0.01 | 0 | 0 |
| `[30,48)` | 0.10 | 0.02 | 0 | 0 |
| `[48,70)` | 0.05 | 0.03 | 0.05 | 0 |
| `[70,90)` | 0.02 | 0.03 | 0.02 | 0.05 |
| `[90,100)` | 0.01 | 0.02 | 0.01 | 0.05 |

The Seam TV coefficient remains `0.004`; the scheduled Seam value is the pair
coefficient used inside the Seam loss. This released 100-epoch, six-stage run
differs from the 110-epoch, three-stage training protocol described in the
paper. Use the table above when reproducing the released Subject 460 run.

The trainer writes the effective configuration to `params.yaml` in the run
directory and stores optimized FLAME parameters in `flame_params.npz` beside
the checkpoint. Both files are required for evaluation and resume behavior.

## Evaluation

The server evaluation script uses `scripts/metrics.sh`. Set its `CKPT_PATH` and
GPU for the target machine, then run:

```bash
bash scripts/metrics.sh
```

Its effective command is:

```bash
CUDA_VISIBLE_DEVICES=1 python metrics.py \
  --checkpoint /path/to/checkpoint_latest.pth \
  --split test
```

`metrics.py` loads `params.yaml` from the checkpoint directory, restores model
and optimized FLAME weights, renders the selected split, and computes the
existing MeGA-aligned metrics.

## Tests

After restoring all authoritative server-side companion files and activating
the `mega` environment, tests can be run with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
bash -n scripts/*.sh stage_train_460_J1J2.sh
```

The release includes the schedule, dataset, smoke-test, comparison, and FLAME
runtime assets copied from the authoritative server tree. Real-batch tests
additionally require the complete MeGA CUDA environment, nvdiffrast, configured
NeRSemble data, and model checkpoints.

## Attribution and licensing

This implementation is based on MeGA and contains FLAME-derived integration
code with its existing copyright notices retained. The downloaded server tree
did not contain a repository-level license file. Before public distribution,
add the project license authorized by the code owners and comply with the
separate terms for MeGA, FLAME, NeRSemble, and all third-party dependencies.
