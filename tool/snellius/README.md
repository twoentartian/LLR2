# Snellius: ytian1 / ViT-B/32 / ImageNet-21k-p

This bundle contains configuration and Slurm submission files. It does not
update the server's training source or automatically submit any jobs.

Server paths:

- Repository: `/gpfs/home6/ytian1/git/LLR2`
- Submission bundle: `/gpfs/home6/ytian1/llr2-jobs`
- Conda environment: `/home/ytian1/miniconda3/envs/torch`
- Input: `/scratch-shared/ytian1/imagenet21k_resized.tar.gz`
- Logs: `/gpfs/home6/ytian1/llr2-jobs/logs`
- Persistent training output: `/gpfs/home6/ytian1/llr2-runs`

## Install dependencies

Run this one command on Snellius (no need to activate the environment):

```bash
/home/ytian1/miniconda3/envs/torch/bin/python -m pip install -r /gpfs/home6/ytian1/llr2-jobs/requirements.txt
```

The requirements match the tested local environment: PyTorch 2.13.0+cu132 /
torchvision 0.28.0+cu132 (CUDA 13.2) and nvidia-dali-cuda130 2.3.0. DALI uses
the cuda130 package name for its CUDA 13 build, including when paired with
CUDA 13.2 PyTorch. The bundle also installs rapidgzip and the imports used by
LLR2's setup package. Source changes, including the corrected ViT presets and
BF16 evaluation, must be transferred via the user's GitHub workflow before
submitting training. Include new files such as `py_src/precision.py` in the
GitHub commit; updating previously tracked files alone is insufficient. The
batch script checks VIT_PRESET_VERSION=2, VIT_LEGACY_PRESET=3, and the presence
of `py_src/precision.py` before extraction, so an incomplete checkout fails early.

## Dataset configuration

The supplied `dataset_env.py` is installed at
`/gpfs/home6/ytian1/git/LLR2/py_src/ml_setup_dataset/dataset_env.py`.
It reads `LLR2_IMAGENET21K_ROOT`, which the batch job sets to an absolute
path inside its node-local `$TMPDIR`. No job ID or temporary path is fixed
in this configuration. Other dataset paths retain their defaults.

The job requests `--constraint=scratch-node` and verifies the resolved
`TMPDIR` is underneath `/scratch-node`. It uses 62 rapidgzip workers with a
64-CPU allocation, streams the decoded TAR into the local NVMe filesystem,
and extracts only `imagenet21k_resized/imagenet21k_train` and
`imagenet21k_resized/imagenet21k_val`. The unused small-class split is skipped.
Both extraction stages must succeed before training begins. Slurm removes
node-local data after the job, so checkpoints and logs go directly to HOME.

## Submit after dependencies and source are ready

The default is 4 H100 GPUs, global batch 512 (128 per GPU), FP16 training AMP,
BF16 validation, DALI, torch.compile, and 90 epochs. The default preset is 3,
the historical ViT recipe: 224-pixel random resized crop, horizontal flip,
ImageNet AutoAugment, Mixup/CutMix (alpha 0.2/1.0), three repetitions, label
smoothing 0.11, no random erasing. DALI therefore processes about three full
training-set passes per epoch. Presets 0/1/2 retain no augmentation / project
v1 / project v2, respectively. Five days is the maximum
walltime requested; the job ends earlier if training finishes. Only the latest
resumable checkpoint is retained, plus final model/optimizer; per-epoch model
archives are disabled. A short CUDA/DALI/NCCL import-and-collective check runs
before the large archive is decoded. It verifies that PyTorch uses CUDA 13.2
and that the CUDA 13 DALI distribution is installed.

```bash
sbatch /gpfs/home6/ytian1/llr2-jobs/vit_b_32_imagenet21k_p.sbatch
```

A scheduling check without submission:

```bash
sbatch --test-only /gpfs/home6/ytian1/llr2-jobs/vit_b_32_imagenet21k_p.sbatch
```

Optional 2-GPU allocation (also halves local NVMe capacity):

```bash
sbatch --gpus-per-node=2 --cpus-per-task=32 /gpfs/home6/ytian1/llr2-jobs/vit_b_32_imagenet21k_p.sbatch
```

One-epoch smoke run, preserving the source recipe's scheduler budget:

```bash
LLR2_EPOCHS=1 sbatch --time=04:00:00 /gpfs/home6/ytian1/llr2-jobs/vit_b_32_imagenet21k_p.sbatch
```

`LLR2_BATCH_SIZE`, `LLR2_SEED`, `LLR2_COMPILE=0/1`, `LLR2_EPOCHS`,
`LLR2_PRESET=0/1/2/3` (default 3), `LLR2_EVAL_PRECISION=bf16/fp32`,
`LLR2_OUTPUT_ROOT`, and `LLR2_DECOMPRESS_THREADS` are optional environment
overrides. Keep the global batch constant when comparing GPU counts.

Resume with the same GPU count as the saved checkpoint; every new allocation
extracts the archive into its own scratch space again:

```bash
LLR2_RESUME_CHECKPOINT=/gpfs/home6/ytian1/llr2-runs/vit_b_32_imagenet21k_p_JOBID/0/training_checkpoint.pt \
  sbatch /gpfs/home6/ytian1/llr2-jobs/vit_b_32_imagenet21k_p.sbatch
```

Resume preserves the checkpoint's evaluation precision unless
LLR2_EVAL_PRECISION is explicitly provided. Recognized pre-fix ViT checkpoints
with unversioned preset 0/1 migrate explicitly to the equivalent preset 3;
the log reports this and the new checkpoint records the original preset.
When an old checkpoint has no evaluation-precision field, resume preserves
historical FP32 evaluation. Versioned preset 0/1/2 checkpoints retain their own
recipes on resume; LLR2_PRESET applies to fresh runs. To compare historical
FP32 validation metrics in a fresh run, set LLR2_EVAL_PRECISION=fp32.

## References

- SURF filesystems: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems
- Parallel gzip decoder: https://github.com/mxmlnkn/rapidgzip
- Official PyTorch wheels: https://pytorch.org/get-started/previous-versions/
- DALI installation: https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/installation.html
