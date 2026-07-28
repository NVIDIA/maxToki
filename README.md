# MaxToki

MaxToki is a temporal AI model for predicting the drivers of cell state progression over time, providing a generalizable framework to decode and control dynamic cellular trajectories. The temporal training is composed of two tasks: 1) predict past, intervening, or future cell states along a trajectory autoregressively (NextCell), and 2) predict the time elapsed between cell state observations as a regression task (TimeBetweenCells). Training uses NeMo and Megatron-LM for distributed GPU execution.

- See [our manuscript](https://www.biorxiv.org/content/10.64898/2026.03.30.715396v1.full.pdf) for details.
- See [the model repository](https://huggingface.co/theodoris-lab/MaxToki) on Hugging Face for the pretrained MaxToki models.

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA A100 | H100 80GB |
| VRAM | 40 GB | 80 GB |
| CUDA | 12.x | 12.4+ |
| Driver | 525+ | latest |

TransformerEngine requires CUDA. The model cannot run on CPU. A single A100 or H100 is sufficient for development and fine-tuning; full-scale pretraining benefits from multiple GPUs.

## Setup

All dependencies (NeMo via PyPI, Megatron-LM, TransformerEngine, Apex) are pinned in the container. Running outside the container is not supported.

### Clone with submodules

Megatron-LM is a git submodule; initialize it with:

```bash
git submodule update --init --recursive
```

### Build the image

```bash
DOCKER_BUILDKIT=1 docker build --target dev -t maxtoki-dev -f Dockerfile .
```

### Launch the container

```bash
docker run --rm -it --gpus all \
  --network host \
  --shm-size=4g \
  -e TMPDIR=/tmp \
  -e NUMBA_CACHE_DIR=/tmp/ \
  -w /workspaces/maxToki \
  -v "$(pwd)":/workspaces/maxToki \
  -v "$HOME/.cache":/home/ubuntu/.cache \
  --user root \
  maxtoki-dev \
  bash -c "usermod -u $(id -u) ubuntu && groupmod -g $(id -g) ubuntu && \
           su - ubuntu -c 'cd /workspaces/maxToki && \
           source .devcontainer/postCreateCommand.sh && exec bash'"
```

This opens a bash shell with the repo mounted at `/workspaces/maxToki` and all bionemo/NeMo sub-packages installed in editable mode. Optionally add `-v /data:/home/ubuntu/data` if you have a local `/data` directory, and pass `-e WANDB_API_KEY=...` for experiment tracking.


## Repository Structure

```
sub-packages/
  bionemo-maxtoki/     # MaxToki model, training, inference, and checkpoint conversion
    src/               # Contains the actual module and source code
    test/              # Contains the tests relevant to maxtoki
  bionemo-llm/         # Shared LLM primitives (Lightning module, LR scheduler, callbacks)
  bionemo-core/        # Core utilities from bionemo-framework
  bionemo-testing/     # General purpose test helpers
3rdparty/
  Megatron-LM/         # Pinned Megatron-LM submodule (NeMo is installed from PyPI)
```

## Architecture

MaxToki is based on the LLaMA decoder model architecture. The first stage pretraining employs an autoregressive training objective to generate rank value encoded transcriptomes using standard cross-entropy loss. In the second stage temporal training, the context length is extended to accommodate an input of multiple single-cell transcriptomes along a cell state trajectory, and the model is trained with a mixed loss (`MaxTokiLossWithReduction`) objective that balances the tasks of cell state generation (cross-entropy loss) and timelapse prediction (MSE loss) using a configurable mixture ratio. 
 
Key classes:

| Class | File | Description |
|---|---|---|
| `MaxTokiBaseConfig` | `api.py` | Base config extending NeMo's `Llama32Config1B` |
| `MaxTokiConfig` | `model.py` | Pretraining config; selects loss class and model class, as well as exposes a variety of transformer-related parameters |
| `MaxTokiMultitaskFineTuneConfig` | `model.py` | Temporal training config; attaches regression head and a custom multitask loss |
| `MaxTokiLossWithReduction` | `model.py` | Mixed CE + MSE loss with per-task masking |
| `MaxTokiTokenizer` | `tokenizer.py` | Wraps the gene token dictionary; handles special tokens and loss mask generation. Generally the tokenizer is a pass-through, rank value encoded token IDs are expected as inputs. |
| `MaxTokiDataModule` | `datamodule.py` | Lightning DataModule for single-cell datasets that may include time tokens. |
| `FinetuneLlamaModel` | `model.py` | MCoreGPTModel subclass with the regression head attached |

## Data Preparation

Raw counts from single-cell RNAseq data (`.h5ad` files) must be processed before training. The pipeline has three stages, all in `bionemo.maxtoki.data_prep`:

**Stage 1: Tokenize** — converts `.h5ad` files into rank value encoding token sequences.

```bash
python -m bionemo.maxtoki.data_prep tokenize \
    --data-directory /path/to/h5ad_files \
    --output-directory /path/to/output \
    --output-prefix my_dataset \
    --nproc 8
```

`--token-dictionary-file` and `--gene-median-file` are required. `--gene-mapping-file` (Ensembl mapping) is optional and only needed when mapping gene names to Ensembl IDs.

**Stage 2: Assemble cell paragraphs** — groups cells from the same trajectory into training sequences that include time-lapse tokens.

```bash
python -m bionemo.maxtoki.data_prep assemble-paragraphs \
    --data-directory /path/to/tokenized.dataset \
    --output-directory /path/to/output \
    --output-prefix my_paragraphs \
    --max-timepoint 730 \
    --time-group-columns donor_id timepoint \
    --num-examples 10000000
```

| Argument | Default | Description |
|---|---|---|
| `--max-timepoint` | required | Maximum time value; sets the numeric range for time tokens. |
| `--time-group-columns` | none | Column names used to group cells into trajectories. |
| `--min-timepoints` | 3 | Minimum observations per paragraph. |
| `--max-timepoints` | 4 | Maximum observations per paragraph. |
| `--task-ratio` | 0.5 | Fraction of samples used for timelapse vs next-cell tasks. |
| `--model-input-size` | 16384 | Sequences longer than this are truncated. |

**Stage 3: Assemble queries** — builds evaluation query datasets from cell paragraphs.

```bash
python -m bionemo.maxtoki.data_prep assemble-queries \
    --blueprint-dictionary-file /path/to/blueprint.pkl \
    --time-token-dictionary-file /path/to/time_dictionary.pkl \
    --cell-paragraph-dataset-file /path/to/paragraphs.dataset \
    --output-directory /path/to/output
```

## Data Format

Datasets are HuggingFace `datasets` directories loaded with `datasets.load_from_disk()`. Each sample has a single field:

- `input_ids`: In the first stage training, input_ids are a rank value encoding of gene token IDs for a single cell transcriptome. In second stage training, input_ids are a sequence of multiple rank value encoded cell transcriptomes as well as tokens representing the time interval between states along the trajectory. The second stage training sequence also includes query structural tokens (`<boq>`, `<eoq>`).

The tokenizer takes a pickle file mapping gene names to integer token IDs. This file also encodes the numeric time-lapse token range.

## Pretraining

Pretraining learns next-token prediction over rank value encoded gene expression sequences. Use `--pretrain` to instruct the datamodule to expect sequences without query structural tokens.

```bash
python -m bionemo.maxtoki.train \
    --train-data-path /path/to/train \
    --val-data-path /path/to/val \
    --test-data-path /path/to/test \
    --tokenizer-path /path/to/token_dictionary.pkl \
    --pretrain \
    --rope-scaling-factor 1.0 \
    --result-dir ./results \
    --experiment-name my_run \
    --num-gpus 1 \
    --num-steps 1000 \
    --lr 1e-4 \
    --seq-length 4096 \
    --micro-batch-size 4 \
    --num-layers 11 \
    --hidden-size 1280 \
    --ffn-hidden-size 2560 \
    --num-attention-heads 8 \
    --rope-scaling-factor 1.0 \
    --output-weights separate \
    --val-check-interval 50 \
    --log-every-n-steps 50
```

Notable arguments:

| Argument | Description |
|---|---|
| `--pretrain` | Switches the datamodule to pretraining mode. Without it, the datamodule expects second-stage data with query structural tokens. |
| `--output-weights` | `tied` shares embedding and LM head weights (one parameter matrix). `separate` gives the LM head its own weights. This is saved in the checkpoint and inherited by all downstream stages. |
| `--rope-scaling-factor` | RoPE frequency scaling. Set to `1.0` for no scaling. Increase to e.g. `4.0` when training on longer sequences. |
| `--seq-length` | Max sequence length, sequences longer than this will be truncated. Shorter sequences have zero-pad tokens added. |
| `--micro-batch-size` | Per-GPU batch size. Global batch size is inferred as `micro_batch_size * num_gpus * num_nodes * accumulate_grad_batches`. |
| `--log-every-n-steps` | How often (in steps) to run the logging workflow. val-check-interval should be a factor of this parameter for consistent checkpointing. |
| `--val-check-interval` | How often (in steps) to run validation and save a checkpoint. Defaults to 10000; clamped to `num_steps` if larger, so short runs only get one checkpoint at the end unless you set this lower. |
| `--result-dir` | Directory in which checkpoints and logs are stored. If it doesnt exist, it will be created |
| `--lr` | Initial learning rate. A cosine schedule ramps up over `--cosine-rampup-frac` of training (default 1%) and decays to `lr/100`. |

Checkpoints are saved under `<result-dir>/<experiment-name>/dev/checkpoints/` with the format `epoch={e}-val_loss={v:.2f}-step={s}-consumed_samples={n}/`. By default the two best checkpoints by `val_loss` are kept, plus the last.

Checkpoint arguments:

| Argument | Default | Description |
|---|---|---|
| `--save-top-k` | 2 | How many best checkpoints to keep. |
| `--save-last-checkpoint` | true | Always keep the most recent checkpoint. |
| `--metric-to-monitor-for-checkpoints` | `val_loss` | Metric used to rank checkpoints. |
| `--disable-checkpointing` | (flag) | Turn off checkpoint saving entirely. |

## Temporal training

Second-stage training adds the TimeBetweenCells regression task on top of continued next-token prediction. Importantly, loss is only calculated for the tokens produced after the <eoq>, representing the model's response to the prompt.
Pass `--use-finetuning-config` to switch to `MaxTokiMultitaskFineTuneConfig`, which attaches the regression head and the mixed loss:

```bash
python -m bionemo.maxtoki.train \
    --train-data-path /path/to/train \
    --val-data-path /path/to/val \
    --test-data-path /path/to/test \
    --tokenizer-path /path/to/token_dictionary.pkl \
    --initial-ckpt-path /path/to/pretrained/checkpoint \
    --result-dir ./results-ft \
    --experiment-name my_ft_run \
    --use-finetuning-config \
    --num-steps 500 \
    --lr 5e-5 \
    --val-check-interval 100 \
    --rope-scaling-factor 4.0 \
    --label-scalar 200.0 \
    --additive-penalty 10.0 \
    --micro-batch-size 4 \
    --limit-val-batches 100 \
    --timelapse-loss mse
```

Notable arguments:

| Argument | Description |
|---|---|
| `--use-finetuning-config` | Switches to `MaxTokiMultitaskFineTuneConfig`. Required for second-stage training. |
| `--initial-ckpt-path` | Path to a pretrained checkpoint. Architecture (num_layers, hidden_size, weight tying) is loaded from the checkpoint; `seq_length` and `rope-scaling-factor` come from CLI args. |
| `--label-scalar` | Regression labels are divided by this before computing loss (default: 200.0). Predictions are multiplied back at inference time. |
| `--additive-penalty` | Added to the regression loss weighted by the probability the model assigns to non-numeric tokens (default: 10.0). Discourages gene tokens from appearing where time tokens are expected. |
| `--timelapse-loss` | Loss function for TimeBetweenCells: `mse` or `ce` over discretized time tokens (default: `mse`). Logged metrics always report MSE regardless. |
| `--freeze-params-until-key-suffix` | Freeze all layers up to and not including the layer whose name ends with this suffix. Useful for freezing early layers. |
| `--limit-val-batches` | The default of 2 batches is usually too few to include TimeBetweenCells samples, so `valid_mse_loss` will show 0. Use 100 or more for meaningful regression metrics. |

## Attention Backend and OOM Notes

The 217M MaxToki model has `hidden_size=1232` and `num_attention_heads=8`, which gives `head_dim=154`. TransformerEngine's Flash Attention kernel requires `head_dim % 8 == 0`. Since `154 % 8 = 2`, TE falls back to an unfused O(N^2) attention implementation. At `seq_length=16384` this needs roughly 33 GB per layer and will OOM on most hardware.

When `--use-finetuning-config` loads a checkpoint with an incompatible head dimension, it automatically switches to PyTorch's SDPA backend.


## Inference

`bionemo.maxtoki.predict` supports two modes. Without `--generate-next-cell`, it runs a single forward pass per sample and outputs the regression prediction for TimeBetweenCells. With `--generate-next-cell`, it autoregressively generates up to `--max-tokens-to-generate` tokens using KV caching.

### Regression (TimeBetweenCells)

```bash
python -m bionemo.maxtoki.predict \
    --initial-ckpt-path /path/to/finetuned/checkpoint \
    --data-path /path/to/predict_data \
    --tokenizer-path /path/to/token_dictionary.pkl \
    --output-dir ./predictions \
    --seq-length 16384
```

This default path is *headless*: the timelapse value is decoded from the LM logits over numeric
tokens. For a model with an **explicit learned regression head** (e.g. one trained with the
HuggingFace code and imported via `import_regression_hf`), add `--regression-head` — see
[HuggingFace regression model to BioNeMo](#huggingface-regression-model-to-bionemo).

### Generation (NextCell)

```bash
python -m bionemo.maxtoki.predict \
    --initial-ckpt-path /path/to/finetuned/checkpoint \
    --data-path /path/to/predict_data \
    --tokenizer-path /path/to/token_dictionary.pkl \
    --output-dir ./predictions \
    --seq-length 16384 \
    --generate-next-cell \
    --max-tokens-to-generate 4096 \
    --top-k 50 \
    --temperature 1.0 \
    --buffer-size-gb 40.0 \
    --buffer-overflow-factor 50.0 
```

Output is written to `predictions__rank_*.pt` files in `--output-dir`. In regression mode each file contains `regression_preds` and `timelapse_token_preds` tensors. In generation mode each file contains a list of dicts with `generated_tokens`, `lengths`, and `full_sequence`.

Arguments common to both modes:

| Argument | Default | Description |
|---|---|---|
| `--initial-ckpt-path` | required | Path to a temporal trained checkpoint. |
| `--data-path` | required | Prediction dataset in HuggingFace `datasets` format. |
| `--tokenizer-path` | required | Token dictionary pickle file. |
| `--output-dir` | required | Where to write prediction files. |
| `--seq-length` | 4096 | Must be at least as long as the longest sequence in the dataset. |
| `--micro-batch-size` | 1 | Per-GPU batch size. |
| `--precision` | `bf16-mixed` | Precision type. |
| `--num-gpus` | 1 | Number of GPUs. |
| `--write-interval` | `epoch` | Write predictions at the end of the epoch (`epoch`) or after each batch (`batch`). |
| `--limit-predict-batches-to-n` | all | Stop after N batches; useful for testing. |
| `--using-pretrain-dataset` | false | Set this if the data uses pretraining format (no `<boq>`/`<eoq>` tokens). |

Generation-only arguments:

| Argument | Default | Description |
|---|---|---|
| `--max-tokens-to-generate` | 4096 | Token budget per sample. |
| `--top-k` | 0 | Top-k sampling; 0 disables it. |
| `--top-p` | 0.0 | Nucleus sampling threshold; 0.0 disables it. |
| `--temperature` | 1.0 | Sampling temperature. |
| `--buffer-size-gb` | 20.0 | KV cache size in GB. |
| `--buffer-overflow-factor` | 50.0 | Controls how tightly the KV cache is packed. 50.0 by defaults allows the full buffer to be used in the KV cache (recommended). |
| `--chunk-size-tokens` | 4096 | Chunk size for KV cache writes. |
| `--naive-benchmarking-only` | false | Skip KV caching entirely. Needed when head_dim is too small for the flash attention decode path. |

## Checkpoint Conversion

### BioNeMo to HuggingFace

`export_hf` loads a checkpoint saved with BioNeMo, maps the weights to Huggingface's `CausalLM`, and saves the corresponding checkpoint.

loads the HF weights into a BioNeMo checkpoint by running one training step with a zero learning rate and saving the result. The output is a standard NeMo checkpoint directory that can be used with `--initial-ckpt-path` in training or `--initial-ckpt-path` in predict.

```bash
python -m bionemo.maxtoki.export_hf \
    --model-path /path/to/bionemo/checkpoint \
    --output-path ./converted_hf \
    --tokenizer-path /path/to/token_dictionary.pkl
```

Pass `--sanity-check` to run a logit comparison between the NeMo and exported HF models after conversion. The script exits non-zero if the mean absolute error exceeds 0.05:

```bash
python -m bionemo.maxtoki.export_hf \
    --model-path /path/to/bionemo/checkpoint \
    --output-path ./converted_hf \
    --tokenizer-path /path/to/token_dictionary.pkl \
    --sanity-check \
    --data-path /path/to/data \
    --num-examples 8
```

### HuggingFace to BioNeMo

`import_hf` loads the HF weights into a BioNeMo checkpoint by running one training step with a zero learning rate and saving the result. The output is a standard NeMo checkpoint directory that can be used with `--initial-ckpt-path` in training or `--initial-ckpt-path` in predict.

```bash
python -m bionemo.maxtoki.import_hf \
    --hf-model-path /path/to/hf/model \
    --train-data-path /path/to/train \
    --val-data-path /path/to/val \
    --test-data-path /path/to/test \
    --tokenizer-path /path/to/token_dictionary.pkl \
    --result-dir ./converted_bionemo \
    --num-gpus 1
```

### HuggingFace regression model to BioNeMo

The `import_hf` path above handles a plain `LlamaForCausalLM` (headless regression). A model
trained with the HuggingFace code that carries an **explicit regression head** is a custom
wrapper instead: a Llama backbone under the `cf.` state-dict prefix (`cf.model.*`,
`cf.lm_head.weight`) plus a `linear1` = `Linear(hidden, 1)` head, and it ships without a
`config.json`. Use `import_regression_hf` for it: it reads the safetensors directly, strips the
`cf.` prefix, reuses the standard backbone mapping, and maps `linear1.* -> regression_head.linear_fc2.*`
on `FinetuneLlamaModel`. The backbone hyperparameters it cannot infer from weight shapes (head
count, RoPE base, norm epsilon) come from a matching Llama `config.json` passed via `--hf-config`
(e.g. `MaxToki-217M-HF/config.json` from the hub); the vocab size is read from the checkpoint.

The import runs in fp32 with the distributed optimizer disabled so the converted weights are
written to the checkpoint verbatim, and runs its single throwaway step on short sequences
(`--compute-seq-length`) while baking the full `--seq-length` into the saved config.

RoPE is taken from the `--hf-config` `rope_scaling` block, not assumed: a `llama3` block maps to its
`factor` and `original_max_position_embeddings`; a `null`/absent block means standard RoPE (no
scaling), which is what the HF MaxToki models use — they handle the extended temporal context by
extrapolation rather than the llama3 scaling that the BioNeMo `train.py` finetune applies
(`--rope-scaling-factor`). Use an `--hf-config` whose `rope_theta`/`rope_scaling` match the trained
model (e.g. `MaxToki-217M-HF/config.json`: θ=10000, `rope_scaling=null`).

#### 1. Convert

```bash
python -m bionemo.maxtoki.import_regression_hf \
    --model-safetensors /path/to/aging_model/model.safetensors \
    --hf-config /path/to/MaxToki-217M-HF/config.json \
    --tokenizer-path /path/to/token_dictionary.json \
    --train-data-path /path/to/heldout.dataset \
    --val-data-path /path/to/heldout.dataset \
    --test-data-path /path/to/heldout.dataset \
    --seq-length 16512 \
    --result-dir ./converted_aging --experiment-name maxtoki_aging
```

This writes a standard BioNeMo checkpoint under
`./converted_aging/maxtoki_aging/dev/checkpoints/<...-last>`.

#### 2. Load the converted model and run inference

The converted checkpoint is a normal BioNeMo checkpoint — load it with `predict` exactly like the
[Regression (TimeBetweenCells)](#regression-timebetweencells) age-prediction path above, just with
the extra `--regression-head` flag. That flag reads the learned head output at the `<eoq>` query
position and rescales by `--label-scalar` (the value the labels were divided by during training; the
aging model used `1000`), instead of decoding the value from the LM logits over numeric tokens. The
prediction dataset must contain only timelapse (`time`) rows, since each must have an `<eoq>` token.
The `<eoq>` id is taken from `--tokenizer-path` by default; pass `--eoq-token-id` only to override it
(e.g. if the tokenizer lacks `<eoq>`).

```bash
python -m bionemo.maxtoki.predict \
    --initial-ckpt-path ./converted_aging/maxtoki_aging/dev/checkpoints/<last-ckpt> \
    --data-path /path/to/heldout_time_rows.dataset \
    --tokenizer-path /path/to/token_dictionary.json \
    --output-dir ./aging_predictions \
    --seq-length 16512 --write-interval batch \
    --regression-head --label-scalar 1000
```

Output lands in the **same format as the age-prediction path**: one
`predictions__rank_*__batch_*.pt` per batch, each a dict with the identical schema both paths use —
`regression_preds` (`[1, B]`, the per-sample timelapse value in real units), `timelapse_token_preds`
(`[1, B]`, the numeric-token argmax at `<eoq>`), and `batch_idx`. Files are ordered by `batch_idx`,
so at `--micro-batch-size 1` on a single GPU they line up 1:1 with the input `time` rows. The
explicit-head path reads `regression_preds` from the learned head and still derives
`timelapse_token_preds` from the LM logits, so the two inference modes are drop-in interchangeable
for any downstream reader.

#### 3. Verify against the reference predictions

To confirm the port reproduces the original HuggingFace predictions, compare against a reference
pickle (a dict with `generated_answer_timelapse` and `ground_truth_answer_timelapse`):

```bash
python -m bionemo.maxtoki.verify_regression \
    --predictions-dir ./aging_predictions \
    --reference /path/to/reference_predictions.pickle
```

On the held-out aging set this reports Pearson ≈ 0.99999 vs the HuggingFace predictions (MAE ≈ 1–3
in months; the small residual is bf16 prediction vs the fp32 reference).

For a copy-pasteable end-to-end runner (subset `time` rows → convert → predict → verify, with all
paths spelled out), see `run-maxtoki-regression-verify.sh` in the repo root.

This whole path is covered by tests: `tests/bionemo/maxtoki/test_regression_import.py` (fast unit
tests for the config/weight mapping and the `<eoq>` gather + `label_scalar` rescale) and
`tests/bionemo/maxtoki/test_regression_conversion.py` (a `slow`, end-to-end import →
`predict --regression-head` → compare-against-the-HF-forward-pass check asserting Pearson > 0.999,
MAE < 5). The end-to-end test needs the external aging artifacts; point it at them with the
`MAXTOKI_AGING_*` env vars documented in that test's module docstring.

## Citation

- J Gόmez Ortega, R D Nadadur, A Kunitomi, S Kothen-Hill, J U G Wagner, S D Kurtoglu, B Kim, M M Reid, T Lu, K Washizu, L Zanders, H Chen, Y Zhang, S Ancheta, S Lichtarge, W A Johnson, C Thompson, D M Phan, A J Combes, A C Yang, N Tadimeti, S Dimmeler, S Yamanaka, M Alexanian, C V Theodoris. Temporal AI model predicts drivers of cell state trajectories across human aging. _**bioRxiv**_, 1 Apr 2026.

