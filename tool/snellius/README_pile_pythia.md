# Pile A/B preparation and Pythia-14M on Snellius

## Data provenance and the meaning of one pass

Use the **original tokenized documents** from
[EleutherAI/pythia_deduped_pile_idxmaps](https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps).
The repository contains 83 binary shards and a GPT-NeoX MMIDIDX index, about 417 GB
including the index. It represents about 207B tokens. The different
[preshuffled repository](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled)
contains the expanded ~300B-token Pythia training sequence, including repeated
presentations. Do not truncate that sequence and assume it is a unique-document pass.
Official near-deduplication does not promise that all remaining text fragments differ.

`tool/prepare_pile.py` downloads no model weights and performs no new text tokenization.
It pins the dataset commit on first download, verifies upstream SHA256 hashes, and
downloads only Pythia's configuration/tokenizer assets, pinned to a model commit too.
The MMIDIDX format and initialization sources are linked in code comments.

## 1. Prepare data on staging

The standalone preparation script and sbatch live outside the server source checkout:

```bash
sbatch /gpfs/home6/ytian1/llr2-jobs/pile/prepare_pile_ab.sbatch
```

Defaults: staging, 4 CPU threads (28 GiB host RAM), no GPU, up to 48 hours,
`LLR2_PILE_ROOT=/scratch-shared/ytian1/pile_deduped`, split seed 42.
Approximately 840 GB of working storage is needed for downloaded source plus outputs;
there is no additional merged source copy. HF caches stay under the dataset root.
Rerun the same command to resume downloads/copying. A different split seed produces
a different `ab_seed<seed>` directory. Existing incompatible outputs are rejected.

The split uses `numpy.random.Generator(PCG64(seed)).permutation(number_of_documents)`:
the first floor(N/2) document IDs belong to A; the rest belong to B. This is a random,
exactly balanced **document-count** split, not a first-half/second-half corpus split.
Token counts need not match exactly. Indexed items belonging to one original document
are kept together. Source order within each selected partition is preserved for fast
sequential I/O; training shuffles sequence indices separately.

Outputs under `ab_seed42/`:

- `A.bin`, `B.bin`: little-endian uint16 tokens; no inserted/deleted EOS or padding.
- `A.doc_offsets.npy`, `B.doc_offsets.npy`: document boundaries in output token units.
- `A.source_document_ids.npy`, `B.source_document_ids.npy`: auditable original IDs.
- `document_partition.npy`: source document -> A=0/B=1.
- `manifest.json`: completion marker, seed, source identity, counts, file hashes.
- `copy_state.json`, `source.doc_offsets.npy`, `identity.json`: preparation/resume metadata.

Only a complete published manifest can be used for training. Copying checkpoints
flush both outputs before publishing the cursor; uncommitted tails are truncated on
resume. Preparation hashes all completed data/index files before publishing.

For an already downloaded original corpus, set `LLR2_PILE_SOURCE_DIR` to its folder.
That skips Hub downloads, including tokenizer assets (the training architecture is
available locally and does not require a tokenizer to read token IDs).

For long-term retention use project/archive storage with sufficient quota. The
[SURF filesystem documentation](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems)
describes automatic expiry on shared scratch (14 days) and deletion of scratch-node
at job end. Merely reading a scratch file does not make it a permanent data store.

## 2. Train from random initialization on four H100s

Sync the new Python files through GitHub before submitting the GPU job:

```bash
sbatch /gpfs/home6/ytian1/llr2-jobs/pile/pythia_14m_pile.sbatch
```

The sbatch is SURF-specific and is intentionally not required in Git. It copies only
prepared A/B outputs and tokenizer assets to scratch-node. There is no Hub access
during GPU training. Persistent run outputs go under `/gpfs/home6/ytian1/llr2-runs/`.

Portable entry point:

```bash
torchrun --standalone --nproc_per_node=4 tool/train_pythia.py \
  --data /path/to/ab_seed42 --output /path/to/run \
  --partition AB --epochs 1 --sequence-length 2048 \
  --micro-batch-size 16 --gradient-accumulation 16
```

The independent causal-LM entry point reuses `py_src.distributed` and leaves the
classifier engine/recipes unchanged. Its data/model modules are reusable for a
subsequent LM misalignment implementation; the existing image misalignment CLI does
not yet accept Pile/Pythia.

Architecture: official Pythia-14M, 14,067,712 parameters, 6 layers, hidden size 128,
4 attention heads, untied 50,304-token embeddings/head, rotary fraction 0.25,
parallel attention/MLP residuals, zero dropout. All weights are newly initialized
AFTER setting the seed. NeoX small_init initializes input/QKV/MLP expansion/final
head weights; wang_init initializes attention/MLP residual output projections.
This follows [the Pythia 14M configuration](https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml)
and [NeoX initialization](https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/init_functions.py),
not Transformers' default fixed initialization standard deviation.

Recipe: AdamW, betas=(0.9,0.95), eps=1e-8, weight decay 0.1 on matrices and zero
on biases/LayerNorm, peak LR 1e-3, minimum 1e-4, clip norm 1, 1% warmup then cosine.
With four GPUs, per-GPU microbatch 16 and accumulation 16 give 1024 sequences /
2,097,152 target tokens per full update, matching Pythia's effective batch.
Adaptations: BF16 instead of original FP16, PyTorch SDPA and optional torch.compile,
and a token-based schedule rescaled to the chosen one-pass budget rather than the
original 143,000 steps / ~300B tokens. `fused=True` is not used.

One pass processes every prediction target once: windows have L+1 input/target
tokens and stride L; the shared input context token is not an extra supervised
target. Short tails are masked, not dropped. Each partition's first token has no
previous context and is not predicted. The two partitions never share context.
DDP does not pad/repeat training examples to fill rank shards; empty rank tails use
all-ignored dummy sequences. Losses/gradients are normalized by actual global valid
target counts across ranks and accumulation, including partial windows.

Default `partition=AB` trains on both partitions. Metrics are token NLL, token
accuracy, and perplexity. A fixed subset is explicitly labeled `training_probe`:
it is **not a held-out validation score**. For a disjoint evaluation use
`LLR2_PILE_PARTITION=A LLR2_EVAL_PARTITION=B` (this trains on A only).

Useful overrides:

```bash
# A short budget/throughput experiment (use the steady-state tokens/s after warmup):
LLR2_MAX_STEPS=100 sbatch /gpfs/home6/ytian1/llr2-jobs/pile/pythia_14m_pile.sbatch

# Resume with the same data, seed, precision, batch, world size, and token budget:
LLR2_RESUME_CHECKPOINT=/path/to/run/training_checkpoint.pt \
  sbatch /gpfs/home6/ytian1/llr2-jobs/pile/pythia_14m_pile.sbatch
```

Checkpoint data include weights, optimizer, per-rank RNG, data fingerprint, epoch,
next global microbatch, token count, and recipe identity. The latest checkpoint is
atomic. Resume reconstructs the same sequence permutation and removes CSV rows
beyond the saved step. Slurm's warning signal requests a coordinated checkpoint.
`summary.json` distinguishes completed token budgets from short benchmark runs.
`initial_model.pt` stores random initialization; `model/` exports our current trained
weights in Hugging Face format. No pretrained model parameter loader is called.

Dependencies in the existing cu13 environment: numpy, huggingface_hub, transformers,
safetensors, and the already installed PyTorch. No DALI or additional attention
extension is required for the Pile entry point.

## Validation

`python -m pytest -q test/test_pile_preparation.py test/test_pythia_pile.py`
checks random document isolation, all-token conservation, interrupted preparation,
token budget/tail handling, seeded architecture construction, exact CPU resume, and
2/4-rank Gloo gradient/optimizer equivalence to one-process training. CUDA BF16/SDPA
and compiled training require a separate GPU smoke check. Full-corpus throughput
and four-H100 NCCL behavior must be measured on SURF; CPU tests do not establish them.
