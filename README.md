# KGENS

Knowledge graph embedding experiments with configurable negative sampling and
filtered link-prediction evaluation. The provided presets cover TransE, DistMult,
ComplEx, and RotatE.

## Installation

Run the following commands from the repository root. Python 3.11 is recommended.
The Python dependencies are NumPy and PyTorch; grouped scatter operations use
native PyTorch functions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`run.sh` requires Bash and enables CUDA for both training and evaluation. Its
examples require an NVIDIA GPU with a working driver. Check that the active
Python environment can access it:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Data

Datasets are read from `kg_data/<dataset>/`. The included dataset directories are
`FB15k237`, `wn18rr`, `YAGO3-10`, and `tree`; names are case-sensitive.

```text
kg_data/
  <dataset>/
    entities.dict
    relations.dict
    train.txt
    valid.txt
    test.txt
```

All files are tab-separated, with no header:

| File | Columns |
| --- | --- |
| `entities.dict` | Integer entity ID, entity name |
| `relations.dict` | Integer relation ID, relation name |
| `train.txt`, `valid.txt`, `test.txt` | Head entity name, relation name, tail entity name |

Entity and relation IDs must be contiguous and start at zero. Every name in a
triple must occur in the corresponding dictionary. Dataset paths containing
`tree` also require `false_negatives.txt`, using the same triple format. These
triples are used for an additional filtered test evaluation.

## Train a model

For example, train TransE on WN18RR using GPU 0 and seed 0:

```bash
bash run.sh train TransE wn18rr 0 0 \
  512 512 250 4.0 6.0 0.0007 100000 8 \
  6.0 0.2 0.0 0.0 "" --valid_steps 5000
```

The wrapper trains the model and evaluates validation and test splits. This
example validates every 5,000 steps; the parser default is 200 steps.

The positional arguments are:

```text
bash run.sh MODE MODEL DATASET GPU SEED \
  BATCH_SIZE NEGATIVES HIDDEN_DIM GAMMA TEMPERATURE LEARNING_RATE MAX_STEPS EVAL_BATCH_SIZE \
  S_GAMMA BETA TAU LAMBDA TAG [EXTRA_ARGS...]
```

| Argument | Meaning |
| --- | --- |
| `MODE` | `train`, `valid`, or `test` |
| `MODEL` | Model name, such as `TransE`, `DistMult`, `ComplEx`, or `RotatE` |
| `DATASET` | Directory name under `kg_data/` |
| `GPU` | CUDA device index |
| `SEED` | Random seed and run identifier |
| `BATCH_SIZE` | Number of positive training triples per batch |
| `NEGATIVES` | Maximum negative-sample allocation |
| `HIDDEN_DIM` | Base embedding dimension |
| `GAMMA` | Scoring margin and embedding initialization parameter |
| `TEMPERATURE` | Adversarial sampling temperature |
| `LEARNING_RATE` | Initial Adam learning rate |
| `MAX_STEPS` | Maximum number of training steps |
| `EVAL_BATCH_SIZE` | Validation/test batch size |
| `S_GAMMA` | Negative-allocation exponent; zero gives equal allocations |
| `BETA` | Negative-count weighting exponent |
| `TAU` | Noise scale for negative allocation; zero disables this noise |
| `LAMBDA` | Interpolation between original and distribution-shape adversarial logits |
| `TAG` | Optional run suffix; pass `""` for no suffix |
| `EXTRA_ARGS` | Additional arguments passed to `codes/run.py` during training |

A nonempty tag is appended literally to the output directory. For example,
`"demo"` adds `_demo`; `"."` adds `_.`.

Use these model-specific extra arguments, as in the preset commands:

| Model | Extra arguments |
| --- | --- |
| TransE | None required |
| DistMult | `--ignore_scoring_margin`; optionally `-r <weight>` |
| ComplEx | `-de -dr --ignore_scoring_margin`; optionally `-r <weight>` |
| RotatE | `-de` |

`-de` doubles the entity embedding dimension and `-dr` doubles the relation
embedding dimension. For example, ComplEx with `HIDDEN_DIM=1000` and `-de -dr`
uses 2,000 real-valued coordinates for each entity and relation embedding.

Other useful training options include `--valid_steps`, `--log_steps`,
`--warm_up_steps`, and `-cpu`. Adversarial negative sampling is enabled by
default; the `-adv` flag disables it.

See all parser options with:

```bash
python codes/run.py --help
```

`best_config.sh`, `best_config_ablation.sh`, and `best_config_others.sh` contain
command collections for the main presets, ablations, and alternative sampling
settings. Select a command and adjust its GPU, seed, or tag for your run.

## Outputs and checkpoints

An untagged run writes to:

```text
models/<MODEL>_<DATASET>/<MODEL>_<DATASET>_<SEED>/
```

For the training example, this is `models/TransE_wn18rr/TransE_wn18rr_0/`.
A tag adds `_<TAG>` to the last directory name.

| File | Contents |
| --- | --- |
| `train.log` | Training losses and validation/test metrics |
| `config.json` | Saved arguments and dataset sizes |
| `checkpoint` | Model and optimizer states, step, and learning-rate schedule state |
| `entity_embedding.npy` | Entity embedding array |
| `relation_embedding.npy` | Relation embedding array |
| `test.log` | Standalone validation/test output |

A checkpoint is saved when validation MRR improves during training. Set
`MAX_STEPS` above the validation interval so at least one scheduled validation
can occur. Final evaluation after training uses the current in-memory model;
standalone evaluation loads the saved checkpoint.

Use a distinct seed or tag for each run you want to retain. Reusing a run
directory overwrites its log and can replace its checkpoint.

## Evaluate a saved model

For the untagged training example, run validation or testing with:

```bash
bash run.sh valid TransE wn18rr 0 0
bash run.sh test TransE wn18rr 0 0
```

These commands load `config.json` and `checkpoint` from the run directory. The
embedding configuration and evaluation batch size are restored from the saved
configuration. Both commands write `test.log`, replacing any previous contents.

For a tagged run, or to select a checkpoint directory explicitly, use the Python
entrypoint from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py \
  --cuda --do_test --data wn18rr \
  --init_checkpoint models/TransE_wn18rr/TransE_wn18rr_0_demo
```

Replace the example checkpoint directory with one created by training. Use
`--do_valid` in place of `--do_test` to evaluate the validation split.

Evaluation predicts both heads and tails and filters known triples from the
training, validation, and test splits. It reports MRR, MR, Hits@1, Hits@3, and
Hits@10, plus directional MRR and Hits@10 by relation cardinality: `1-to-1`,
`1-to-N`, `N-to-1`, and `N-to-N`. Categories with no evaluated queries have `NaN`
scores. Relations are categorized using the training split.

## Reproducibility

The training entrypoint seeds Python, NumPy, PyTorch, and data loading. It enables
deterministic PyTorch algorithms and disables cuDNN benchmarking and TF32. The
native scatter module also enables deterministic algorithms when imported.
Repeatability should be assessed with the same hardware and software versions;
results can differ across environments.

Generated bytecode, local environments, logs, and model outputs are excluded
from Git by `.gitignore`.

## Acknowledgements

We thank the authors of the following works for their contributions to knowledge
graph embedding and negative sampling:

- [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/pdf/1902.10197)
  (2019).
- [Entity-Relation Distribution-Aware Negative Sampling for Knowledge Graph Embedding](https://link.springer.com/chapter/10.1007/978-3-031-47240-4_13)
  (2023).
