# OpenFWI Diffusion Inversion

A YAML-driven project for training an unconditional diffusion prior and running two waveform-inversion workflows on OpenFWI velocity models:

- `diffusion`: diffusion sampling guided by a Deepwave data-misfit gradient;
- `train`: DDPM training on NumPy velocity models.

Dataset choice and experiment values are centralized in one YAML file. Each invocation runs one configuration; there are no nested parameter-sweep loops in the entry points.

## Project layout

```text
openfwi-inversion/
├── configs/
│   └── config.yaml
├── src/openfwi_inversion/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── diffusion_core.py
│   ├── diffusion_runner.py
│   ├── optimization_core.py
│   ├── optimization_runner.py
│   ├── training.py
│   └── training_core.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended. Deepwave compilation and supported CUDA versions depend on the installed PyTorch build.

```bash
cd openfwi-inversion
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a platform-specific CUDA build, install PyTorch using the command from the PyTorch website before running `pip install -e .`.

## Data and models

Download velocity-model labels from the [official OpenFWI dataset page](https://openfwi-lanl.github.io/docs/data.html). The listed Vel and Fault datasets use `70 x 70` target velocity models. OpenFWI files commonly contain a batch shaped like `(500, 1, 70, 70)`; the training loader accepts these batch files directly, as well as individual `(70, 70)` or `(1, 70, 70)` files.

If another workflow requires one model per file, split a downloaded batch along
its first dimension with:

```bash
openfwi split data/FlatFault_A/model1.npy data/FlatFault_A/split
```

For an input shaped `(500, 1, 70, 70)`, this writes 500 files named
`model1_00000.npy` through `model1_00499.npy`, each shaped `(1, 70, 70)`.
Use `--samples-per-file 50` to create 10 smaller batch files instead, or
`--overwrite` to replace an earlier split. The same operation is available in
Python:

```python
from openfwi_inversion.data import split_velocity_batch

files = split_velocity_batch("data/FlatFault_A/model1.npy", "data/FlatFault_A/split")
```

When training from the split files, point that dataset profile's `train_dir` at
the `split` directory. Do not include both the original batch and its split
copies in the same training file glob, or the same samples will be loaded twice.

A suggested local layout is:

```text
openfwi-inversion/
├── data/
│   ├── FlatFault_A/
│   ├── FlatFault_B/
│   ├── CurveFault_A/
│   ├── CurveFault_B/
│   ├── CurveVel_A/
│   └── CurveVel_B/
├── models/
│   └── FlatFault-B/
│       ├── unet/
│       └── scheduler/
└── ...
```

Diffusion inversion requires a Diffusers model directory containing `unet/` and `scheduler/`. Train one with this project or point `model_path` to an existing compatible checkpoint.

### Pretrained diffusion models

Pretrained diffusion priors for all six OpenFWI model families are available from the [OpenFWI Diffusion Priors repository on Hugging Face](https://huggingface.co/PositivePassion/openfwi-diffusion-priors):

| Model family | Files |
| --- | --- |
| FlatFault A | [`FlatFault-A`](https://huggingface.co/PositivePassion/openfwi-diffusion-priors/tree/main/FlatFault-A) |
| FlatFault B | [`FlatFault-B`](https://huggingface.co/PositivePassion/openfwi-diffusion-priors/tree/main/FlatFault-B) |
| CurveFault A | [`CurveFault-A`](https://huggingface.co/PositivePassion/openfwi-diffusion-priors/tree/main/CurveFault-A) |
| CurveFault B | [`CurveFault-B`](https://huggingface.co/PositivePassion/openfwi-diffusion-priors/tree/main/CurveFault-B) |
| CurveVel A | [`CurveVel-A`](https://huggingface.co/PositivePassion/openfwi-diffusion-priors/tree/main/CurveVel-A) |
| CurveVel B | [`CurveVel-B`](https://huggingface.co/PositivePassion/openfwi-diffusion-priors/tree/main/CurveVel-B) |

From the project root, install the Hugging Face CLI and download all models into the default location used by `configs/config.yaml`:

```bash
python -m pip install -U huggingface_hub
hf download PositivePassion/openfwi-diffusion-priors --local-dir models
```

To download only one model family, use an include pattern:

```bash
hf download PositivePassion/openfwi-diffusion-priors \
  --include "CurveFault-A/*" \
  --local-dir models
```

The model repository includes `SHA256SUMS` for checking the six weight files after download.

All paths in the YAML file are resolved relative to the YAML file itself, not relative to the current shell directory. Update the example paths after downloading the data.

## Configuration

Edit `configs/config.yaml` before running. The important top-level selectors are:

```yaml
mode: diffusion       # diffusion, optimization, or train
dataset: flat_fault_b # one key under datasets
```

Each dataset profile has the same interface:

```yaml
datasets:
  my_dataset:
    train_dir: ../data/MyDataset
    sample_file: ../data/MyDataset/model1.npy
    sample_index: 0
    model_path: ../models/MyDataset
```

Add another profile in the same form to use a new dataset; no Python changes are required. `train_dir` is used by training, `sample_file` by inversion, and `model_path` by diffusion inversion or as the default training architecture source. Set `sample_index` for a batched OpenFWI label file, or set it to `null` when `sample_file` already contains one model.

For simultaneous diffusion inversion, replace `sample_file` with a `samples` list:

```yaml
datasets:
  curve_fault_a_batch:
    train_dir: ../data/CurveFault_A
    model_path: ../models/CurveFault-A
    samples:
      - name: ex9
        file: ../data/CurveFault_A/model1.npy
        index: 9
      - name: ex19
        file: ../data/CurveFault_A/model1.npy
        index: 19
```

The batch size is inferred from this list. The UNet and Deepwave propagation run with a real batch dimension; losses, errors, tensors, and plots are also saved separately for every sample.

To invert one model with several seeds in the same batch, repeat its `file`
entry with different `name` and `seed` values under `samples`. Each entry then
uses independent random streams for its initial latent, DDPM steps, and
observation noise. Entries without `seed` use `runtime.seed` when any entry
has an explicit seed. Without per-entry seeds, the existing shared batch
random stream is preserved. Per-entry seeds are saved in `metrics.json`.

The remaining sections control normalization, device and seed, acquisition geometry, and workflow-specific settings. To run several experiments, create several YAML files and launch each explicitly. Lists of parameter values are intentionally not expanded into sweeps.

## Run

From the project root:

```bash
openfwi --config configs/config.yaml
```

The module form is equivalent:

```bash
python -m openfwi_inversion.cli --config configs/config.yaml
```

Results are written under:

```text
outputs/<dataset>/<mode>/
```

Each output directory contains a copy of the input `config.yaml` for reproducibility. Inversion outputs also include the final tensor, comparison and history plots, and `metrics.json`. Training outputs include TensorBoard logs, evaluation samples, and Diffusers-compatible epoch checkpoints.

To inspect training progress:

```bash
tensorboard --logdir outputs/<dataset>/train/logs
```

## Notes

- The default acquisition assumes a `70 x 70` velocity model and pads diffusion-training samples to `72 x 72`.
- `train.file_glob` defaults to `**/model*.npy` so that downloaded seismic-data arrays are not treated as velocity labels. Adjust it to match renamed or preprocessed files.
- Set `runtime.device` to the desired device, for example `cuda:0` or `cpu`. Full inversion is computationally expensive on CPU.
- Direct inversion uses GD with gradient scaling and velocity bounds; set `optimization.learning_rate` to control the step size.
- `diffusion.loss` accepts `mse` or `w2`; the direct optimization workflow currently uses its W2 objective.
- Set `train.resume_dir` to an `epoch_XXXX` checkpoint or a parent directory containing epoch checkpoints to resume training.
