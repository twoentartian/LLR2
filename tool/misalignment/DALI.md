# Misalignment ImageNet loading

`misalignment_measurement_2.py` and `misalignment_cct_entry.py` accept
`--loader_backend auto|pytorch|dali`. The default `auto` selects DALI for
ImageNet-1k and PyTorch for MNIST, CIFAR-10/100 and modular data. Existing
ImageNet launch scripts need no extra flag. Use `--loader_backend pytorch`
to compare with the previous loader or run ImageNet on CPU. DALI requires
CUDA and a working NVIDIA DALI installation in the executing Python
environment; missing DALI raises an error rather than silently falling back.
The implementation is tested with DALI 2.3.0.

The original ImageFolder datasets are still combined, shuffled and divided
into equal halves first. DALI reads each half's explicit file/label manifest
in that exact order. No image files are copied, and the complete decoded
dataset is not cached in RAM or VRAM. Only bounded batches are prefetched.
Padded final batches are trimmed, including singleton batches: no real
examples are dropped or counted twice. No repeated augmentation, Mixup,
CutMix or extra per-epoch shuffle is introduced.

Both halves use the same augmentation level, regardless of their train/val
role. This also applies to evaluation and relative flatness, matching the
experiment's existing augmentation policy:

- `none/0`: mixed JPEG decode, GPU bilinear resize and center crop, normalize.
- `1`: mixed random-crop decode, GPU resize, horizontal flip, normalize.
- `2`: level 1 plus TrivialAugmentWide, RandAugment (2 operations, magnitude 9)
  and random erasing (p=0.1) before normalization. Erase rectangles use ten
  rejection attempts rather than clamping an invalid rectangle to the image.

The input size comes from the augmentation factory: CCT's launcher keeps
224 for every ImageNet level; the generic preset 2 retains 176. DALI uses
its native augmentation operators. JPEG decoding, crop sampling, rounding,
resizing and augmentation RNG are **not bitwise equivalent** to torchvision;
switching the backend can change trajectories. Keep the backend fixed when
comparing experiments. Partition-specific RNG seeds are reset per traversal
and depend on the augmentation epoch, not the train/val role. This makes
repeated evaluation and swapped directions reproducible for a fixed loader
configuration; it does not promise batch-size-independent augmentations.

`--num_workers` is the DALI CPU thread count per pipeline (minimum 1), not
a PyTorch multiprocessing worker count. `--prefetch_factor` is DALI's
pipeline prefetch depth (default 4), rather than batches per worker. JPEG
decode uses `mixed` CPU/GPU execution, while resize/augmentation/normalize
run on GPU; only tiny random-erasing rectangle parameters are generated
on CPU. Training/evaluation use the existing AMP configuration. Flatness
remains full-precision, with the existing microbatch size and sample count.
Training pipeline pools are released before flatness; an interrupted pass
is discarded so the next checkpoint measurement restarts at the first image.

The log reports the chosen backend. `summary.json` records the backend,
DALI version, threads, prefetch depth, input size and seed policy for each
training/evaluation and flatness loader.

Run the regression suite with:

```sh
python -m unittest discover -s test -p 'test_misalignment*.py' -v
```

GPU integration tests cover all augmentation levels, exact sample counts,
singleton tails, interrupted/repeated passes, epoch rewind, AMP evaluation,
both orientations, saved gap checkpoints and actual Hessian measurements.
They are skipped when DALI or CUDA is absent; metadata/PyTorch tests still run.
