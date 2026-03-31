# MaxToki

MaxToki is a temporal AI model for predicting the drivers of cell state progression over time, providing a generalizable framework to decode and control dynamic cellular trajectories. The temporal training is composed of two tasks: 1) predict past, intervening, or future cell states along a trajectory autoregressively (NextCell), and 2) predict the time elapsed between cell state observations as a regression task (TimeBetweenCells). Training uses NeMo and Megatron-LM for distributed GPU execution.

- See [our manuscript](https://www.biorxiv.org/content/xxxx.pdf) for details.
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

All dependencies (NeMo, Megatron-LM, TransformerEngine, Apex) are pinned in the container. Running outside the container is not supported.

### Clone with submodules

NeMo and Megatron-LM are git submodules; initialize them with:

```bash
git submodule update --init --recursive
```

### Build the image

```bash
DOCKER_BUILDKIT=1 docker build --target dev -t maxtoki-dev -f Dockerfile .
```

### Launch the container

(!! NOTE can we expand and keep from using internal scripts here? )
```bash
./internal/scripts/run_dev.sh
```

This builds the image if not cached, then opens a bash shell with the repo mounted at `/workspaces/bionemo-framework-private`. Force a rebuild with `REBUILD=1`.

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
  NeMo/                # Pinned NeMo submodule
  Megatron-LM/         # Pinned Megatron-LM submodule
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
(!! what about collation?)
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

Resource files (token dictionary, gene median, Ensembl mapping) default to the bundled versions in `bionemo/maxtoki/data_prep/resources/`. Pass `--token-dictionary-file`, `--gene-median-file`, or `--gene-mapping-file` to override.

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

## Citation

- J Gόmez Ortega, R D Nadadur, A Kunitomi, S Kothen-Hill, J U G Wagner, S D Kurtoglu, B Kim, M M Reid, T Lu, K Washizu, L Zanders, H Chen, Y Zhang, S Ancheta, S Lichtarge, W A Johnson, C Thompson, D M Phan, A J Combes, A C Yang, N Tadimeti, S Dimmeler, S Yamanaka, M Alexanian, C V Theodoris. Temporal AI model predicts drivers of cell state trajectories across human aging. _**bioRxiv**_, 1 Apr 2026.

`import_hf` loads the HF weights into a BioNeMo checkpoint by running one training step with a zero learning rate and saving the result. The output is a standard NeMo checkpoint directory that can be used with `--initial-ckpt-path` in training or `--initial-ckpt-path` in predict.
