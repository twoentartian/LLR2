# ConvNeXt-Tiny / ImageNet-21k-p on Snellius

This is a **random-initialization** experiment. No pretrained weights are loaded
or downloaded. Sources are updated through the user's GitHub workflow; only
this sbatch/configuration bundle is copied directly to SURF.

Submit after committing **all new files** and syncing source to SURF:

```bash
sbatch /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p.sbatch
```

To explicitly clear an exported resume path for a fresh run:

```bash
LLR2_RESUME_CHECKPOINT= sbatch /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p.sbatch
```

Defaults: one node, 4 H100 GPUs, 64 CPUs, CUDA 13.2 PyTorch, DALI, FP16 training
AMP, BF16 validation, torch.compile, seed 42, 90 epochs / 5 warmup. Global
microbatch 512 (128/GPU), accumulation 8, effective batch 4096, peak LR 0.004,
minimum LR 0.000001, AdamW. The scheduler counts optimizer updates, including
the final short accumulation group. Changing batch/accumulation linearly scales
both LR endpoints from the reference effective batch 4096; this is an explicit
adaptation rather than an independently validated recipe.

The default `LLR2_PRESET=4` is the ConvNeXt recipe; preset 3 retains the complete
historical ViT augmentation, including DALI RA3. See the detailed comparison in
[the augmentation reference](../../py_src/ml_setup_dataset/README_imagenet_augmentation.md).
Preset 4 uses 224 input, bicubic crop/resize, RandAugment
`rand-m9-mstd0.5-inc1`, Mixup 0.8 / CutMix 1.0, target smoothing 0.1, pixel
random erasing 0.25, and **one dataset pass per epoch**. Tiny stochastic depth
is zero; EMA and gradient clipping are disabled, following the 22k recipe.
DALI is a GPU adaptation of the policy, not a pixel-identical PIL backend.

The ImageNet-21k-p head follows the actual class directories (normally 10,450).
LayerNorm, biases and LayerScale have zero weight decay; other parameters use
0.05. Worker re-initialization preserves ConvNeXt's truncated-normal weights
(std 0.02) and LayerScale 1e-6 instead of generic Linear/Conv reset rules.

The shared environment, dataset override, rapidgzip staging to node-local
Scratch and persistent outputs follow [the existing SURF guide](README.md).
No additional packages are needed beyond that environment (`timm` is included).
Fresh output is `/gpfs/home6/ytian1/llr2-runs/convnext_tiny_imagenet21k_p_JOBID`.
The batch script checks the new source/recipe files **before extraction**.

Optional overrides:

```bash
# One-epoch smoke run; optimizer schedule still has the 90-epoch recipe budget.
LLR2_EPOCHS=1 sbatch --time=04:00:00 /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p.sbatch

# Keep the effective batch unchanged with larger microbatches.
LLR2_BATCH_SIZE=1024 LLR2_ACCUMULATION=4 sbatch /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p.sbatch

# Explicit historical augmentation control; this triples DALI data passes.
LLR2_PRESET=3 sbatch /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p.sbatch

# Resume with the SAME GPU count. Saved batch, accumulation and preset are restored.
LLR2_RESUME_CHECKPOINT=/gpfs/home6/ytian1/llr2-runs/convnext_tiny_imagenet21k_p_JOBID/0/training_checkpoint.pt \
  sbatch /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p.sbatch
```

`LLR2_EVAL_PRECISION=fp32/bf16`, `LLR2_COMPILE=0/1`, `LLR2_SEED`,
`LLR2_DECOMPRESS_THREADS`, and output/environment overrides match the ViT bundle.
`sbatch --test-only SCRIPT` checks scheduling without submitting training.

## Misalignment

`convnext_tiny_imagenet21k_p_misalignment.sbatch` runs
`tool/misalignment_measurement_2.py` with the same optimizer recipe: global
paired microbatch 512 per partition, `--gradient_accumulate_every 8`
(effective 4096; the script refuses other products), AdamW peak LR 0.004,
wd 0.05 with the same no-decay groups, 90 epochs / 5 warmup, cosine to 1e-6.
Raw A/B gradients of the 8 microbatches are averaged at one parameter state
before normalized combination, so each update matches a 4096 batch. No
augmentation, `--model_preset 4` (hard-label CE: Mixup's smoothing is absent),
BF16 AMP, train/val gradient weights 0.8/0.2, seed 1, split seed 2026,
evaluation in the last 9 epochs. Both directions run in one job by default
(`LLR2_DIRECTIONS=a_as_train_b_as_val` selects one); flatness is deferred.

```bash
sbatch /gpfs/home6/ytian1/llr2-jobs/convnext_tiny_imagenet21k_p_misalignment.sbatch
```

References: [ConvNeXt paper, Table 5](https://arxiv.org/html/2201.03545v2#A1.SS1),
[official training](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md),
[official augmentation defaults](https://github.com/facebookresearch/ConvNeXt/blob/main/main.py).
The source reference is full ImageNet-22k; using the processed winter21 21k-p
subset is a dataset adaptation, not a claim of exact paper reproduction.
