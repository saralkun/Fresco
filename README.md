# Fresco

**Frequency-Spatial Consistent Optimization for Fine-Grained Head Avatar Modeling**  
CVPR 2026

This repository contains the official implementation of Fresco for training and evaluating dynamic head avatars on NeRSemble. Fresco combines image-space frequency supervision with UV-space consistency on a hybrid mesh¨CGaussian representation.

## Setup and Data

Follow the [MeGA setup instructions](https://github.com/conallwang/MeGA#environment) to configure the Python and CUDA environment. Download the preprocessed NeRSemble data for your chosen subject from [MeGA's data release](https://github.com/conallwang/MeGA#data--pretrained-models).

Run the commands below from the repository root.

## Training

We use NeRSemble Subject 460 as an example. Set the dataset and model asset paths in `configs/nersemble/460/full.yaml`, then run:

```bash
conda activate mega
GPU_ID=0 NUM_WORKERS=2 bash ./scripts/train_fresco_460_100ep.sh --run
```

Use `--dry-run` to preview the launch command and training schedule. The script resumes from `checkpoint_latest.pth` if one exists in the run directory. Set a new `RUN_NAME` to start a fresh run.

For another subject, prepare a configuration with the corresponding data and initialization assets, then specify its path with `CONFIG_PATH` and use a separate `RUN_NAME`.



## Evaluation

Set the checkpoint path, split (`val` or `test`), and GPU in `scripts/metrics.sh`. Keep `params.yaml` and `flame_params.npz` from the same training run alongside the checkpoint.

```bash
bash ./scripts/metrics.sh
```

The script renders the selected split and computes image-quality metrics.

## Acknowledgements

FReSCO builds on the [MeGA](https://github.com/conallwang/MeGA). We thank the MeGA authors for sharing their code and the preprocessed NeRSemble data used for training and evaluation, and the NeRSemble authors for providing the original dataset.
