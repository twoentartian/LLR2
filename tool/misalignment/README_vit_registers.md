# ViT-B/32 register comparison

Model names are `vit_b_32` and `vit_b_32_reg4`. Both are randomly initialized
224px ViT-B/32 classifiers. The latter adds exactly four learnable tokens
(3,072 parameters), after positional embeddings have been added to CLS and
patch tokens. Registers have no positional embeddings, participate in every
attention layer, and their final outputs are not used by the classifier.

The original backbone parameter names, shapes and initialization are preserved.
With the same `--random_seed`, both models start with identical common weights;
register initialization consumes random numbers only after the backbone is
initialized. Use the same explicit `--split_seed`, batch size and training
options for comparisons. Checkpoints include the new `vit_b_32_reg4` model name
and can be reconstructed through `get_ml_setup_from_config`.

## Running

Run from the repository root. This example supplies optimizer and scheduler
options explicitly; the measurement update rule remains controlled by its
existing CLI arguments. The 20-epoch budget and batch size 128 are starting
points, not measured runtime or memory guarantees.

```bash
for model in vit_b_32 vit_b_32_reg4; do
  python tool/misalignment_measurement_2.py \
    --dataset imagenet1k --model_type "$model" \
    --random_seed 0 --split_seed 2026 --directions both \
    --augmentation none --batch_size 128 --epochs 20 \
    --optimizer adamw --learning_rate 1e-4 --weight_decay 0.01 \
    --scheduler fixed --warmup_epochs 2 \
    --relative_flatness_layers last_matrix_weight \
    --relative_flatness_batch_size 32 \
    --relative_flatness_batches 16 --relative_flatness_samples 8 \
    --token_norm_threshold 150 \
    -o "results/${model}_seed0"
done
```

The inherited ViT classification loss uses label smoothing 0.11 in both models.
The measurement script applies its own per-image augmentation; the MLSetup's
Mixup/CutMix collate and repeated-augmentation sampler are not used here.
Augmentation defaults to `none` in the measurement script. Its existing generic
ImageNet augmentation level 2 uses a 176px crop, which is incompatible with
these fixed-224px models; use `none` or level 1 unless overriding crop size.

## Norm output

Each training epoch displays a `tqdm` bar in an interactive terminal, with the
direction, epoch number, completed batches, processing rate (`batch/s`) and ETA.
One batch here means one paired A/B optimizer update. Use `--progress` to force
display when output is redirected, or `--no-progress` to disable it. Separate
`eval train` and `eval val` bars show full-partition evaluation after training.
The epoch log and optimization CSV record `train_seconds`, `eval_train_seconds`,
`eval_val_seconds`, `checkpoint_seconds` and `epoch_seconds`. The symmetric
objectives also report `baseline_eval_seconds`.

ViT norm measurement is enabled automatically. Use `--no-token_norms` to disable
it, or `--token_norm_threshold VALUE` to change the reporting threshold. This
threshold is a diagnostic choice, not a universal definition of artifacts.

Each direction writes `token_norms_<direction>.csv`. Every epoch has separate
train/val rows for `patch`, `cls`, and (in the register model) `register_0` through
`register_3`. Columns are:

- `count`: number of tokens across the complete partition, including the last
  short batch. Special tokens are excluded from the patch group.
- `mean`, `std`, `max`: L2 norm moments and maximum. Standard deviation uses
  population normalization, not an average of per-batch standard deviations.
- `high_norm_ratio`: fraction whose norm is strictly greater than the threshold.
- `histogram_counts`: JSON integer list, aligned to histogram edges saved in
  `summary.json` under each direction's `token_norm_diagnostics`. Bins are
  `[0, first_edge)`, `[edge_i, edge_i+1)`, and `[last_edge, infinity)`.

Norms are observed at `encoder.ln.input`, after the last transformer block and
before final LayerNorm. Computation uses detached FP32 tokens. Streaming moments
and fixed histograms reuse epoch evaluation forwards, without saving feature
maps or adding extra model forwards. The monitor is inactive during training
and second-order flatness passes. Final train/val norm summaries are also saved
in `summary.json`. Other model families retain their existing evaluation path.

This records final-layer token norms, not per-layer norms or attention maps.
The existing `misalignment_measure` definition and gradient update rules are
unchanged.

## Shared DALI preprocessing

The misalignment and high-accuracy generators both call
`py_src.ml_setup_dataset.dataset_imagenet.build_dali_imagenet_pipeline`.
The image operations are extracted from the high-accuracy generator's existing
implementation: CPU decode, followed by GPU resize and normalization. With
`--augmentation none`, both partitions use its deterministic evaluation recipe
(short edge 256 and center crop 224 for these ViTs). Levels 1 and 2 select its
basic and advanced training recipes, with the configured crop size.

The misalignment adapter supplies explicit A/B file/label manifests, fixed
reader order, and an independent pass seed. Mixup/CutMix and repeated dataset
passes are not enabled. JPEG fancy upsampling, explicit resize antialiasing and
the separate Python random-erase implementation from the former misalignment
pipeline have been removed in favor of the shared image operations. Result
metadata records `decoder_device=cpu` and the shared preprocessing source.

## Evaluation throughput

Use `--eval_num_workers` to tune evaluation decoding independently of training,
and `--eval_batch_size` to set its inference batch size. Both default to the
training values. Loader workers default to 8 (`--num_workers 8`); DALI uses
this as the decoder thread count per pipeline. Full A/B partitions and token
norm statistics are still
measured every epoch; evaluation has no subsampling or skipped epochs.
Final metrics reuse the last epoch's evaluation instead of repeating both
complete passes after training.

On the RTX 5090 server, an 8,192-image probe at batch size 128 improved from
about 3,360 images/s with 4 decoder threads to 5,266 images/s with 12 threads.
Keeping batch size 128 preserved loss, accuracy and norm statistics exactly
in this probe. Batch sizes 256/512 were faster but introduced small BF16
rounding differences, so the conservative setting is:

```bash
--batch_size 128 --num_workers 4 --eval_batch_size 128 --eval_num_workers 12
```

A full partition check with 665,583 images at batch size 128 and 12 threads
took 125.1 seconds (5,322 images/s). Loss, accuracy and all token norm statistics
matched the saved run exactly. With the original training loop taking about
350 seconds, two evaluation passes at this rate would put an epoch near
10 minutes; a complete training epoch with these settings has not been timed.
