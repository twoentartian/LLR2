# Archived tools

These scripts are kept for repairing or reproducing historical results rather
than as regular experiment entry points.

## remeasure_misalignment_flatness.py

Re-measures saved final models and gap checkpoints on deterministic,
unaugmented data to correct historical flatness measurements made on random
augmented views. It preserves the original models and results, writing to a
new subdirectory.

Run from the repository root:

```bash
python tool/archive/remeasure_misalignment_flatness.py /path/to/old/result
```

The original summary must contain a non-null `relative_flatness` configuration;
this script reuses that measurement budget. New experiments already use
unaugmented flatness loaders. For a deferred measurement of a single checkpoint,
use `tool/misalignment_measurement_2.py --flatness_from_checkpoint ...` with the
original loss and split settings.
