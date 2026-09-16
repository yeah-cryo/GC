# Experiment Runbook

This file lists every executable workflow in `scripts/` and `tools/`, grouped by use case. Run commands from the repository root:

```bash
cd /mnt/e/repos/real_reference
```

Shell wrappers use `.venv/bin/python`, set `PYTHONPATH=src`, and capture logs where applicable. To use another interpreter, set `PYTHON=/path/to/python`. Training and generation require a CUDA GPU with BF16 support. Set `SWANLAB_API_KEY` in the environment for workflows configured to log to SwanLab; do not store the key in the repository.

Long jobs can be kept alive in tmux. For example:

```bash
tmux new-session -d -s JOB_NAME \
  'cd /mnt/e/repos/real_reference && bash scripts/COMMAND.sh ARGS'
```

## 1. Prepare COCO captions

### Export the first 20,000 COCO 2014 training captions

Purpose: read the official COCO caption annotations in their stored order and save a reproducible JSONL prompt manifest containing `prompt_index`, `annotation_id`, `image_id`, and `caption`.

Entry point: `tools/prepare_coco_prompts.py`

```bash
PYTHONPATH=src .venv/bin/python tools/prepare_coco_prompts.py \
  --annotations /path/to/annotations_trainval2014.zip \
  --output outputs/coco2014_prompts/first_20000.jsonl \
  --count 20000
```

`--annotations` may also point directly to `captions_train2014.json`. The command writes a metadata file beside the manifest with the source hash and selection policy. The required 20,000-prompt manifest is already present at `outputs/coco2014_prompts/first_20000.jsonl`.

## 2. Train the baseline ResNet-50 critic

### Start baseline training

Purpose: fine-tune the full ImageNet-pretrained ResNet-50 backbone and one-logit binary head on the GenImage SD1.4 training split. It creates the deterministic 95/5 training/internal-validation manifests and writes checkpoints, metrics, logs, and SwanLab metadata.

Wrapper: `scripts/train.sh`  
Entry point: `tools/train_critic.py`  
Default config: `configs/experiments/resnet50_critic.yaml`

```bash
bash scripts/train.sh
```

Resume from the last completed epoch:

```bash
bash scripts/train.sh \
  --resume outputs/resnet50_critic/latest.pt
```

Override the main paths without editing the YAML:

```bash
bash scripts/train.sh \
  --data-root /mnt/f/datasets/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4 \
  --pretrained models/resnet50-11ad3fa6.pth \
  --output outputs/resnet50_critic_new
```

Direct Python equivalent:

```bash
PYTHONPATH=src .venv/bin/python -u tools/train_critic.py \
  --config configs/experiments/resnet50_critic.yaml
```

## 3. Evaluate a ResNet-50 critic

### Evaluate one official GenImage validation split

Purpose: evaluate a checkpoint on one `val/{nature,ai}` split using fake-image JPEG quality 96, five fixed crops, averaged logits, and balanced accuracy.

Wrapper: `scripts/evaluate.sh`  
Entry point: `tools/evaluate_critic.py`

```bash
bash scripts/evaluate.sh \
  --checkpoint outputs/resnet50_critic/best.pt \
  --data-root /mnt/f/datasets/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4 \
  --output outputs/resnet50_critic/genimage_evaluation/stable_diffusion_v_1_4.json
```

Optional arguments are `--batch-size`, `--workers`, and `--device`. If `--output` is omitted, results go to `official_val_metrics.json` beside the checkpoint.

### Evaluate all unseen GenImage generators

Purpose: evaluate ADM, BigGAN, Midjourney, VQDM, GLIDE, SD1.5, and Wukong, producing per-generator JSON files plus `summary.json` and `summary.csv`. SD1.4 is evaluated separately with `scripts/evaluate.sh`.

Wrapper: `scripts/evaluate_genimage.sh`  
Entry point: `tools/evaluate_genimage.py`

Baseline critic:

```bash
bash scripts/evaluate_genimage.sh \
  --checkpoint outputs/resnet50_critic/best.pt \
  --data-root /mnt/f/datasets/GenImage \
  --output outputs/resnet50_critic/genimage_evaluation
```

PROBE-fine-tuned critic:

```bash
bash scripts/evaluate_genimage.sh \
  --checkpoint outputs/resnet50_critic_probe_guided_s20/best.pt \
  --data-root /mnt/f/datasets/GenImage \
  --output outputs/resnet50_critic_probe_guided_s20/genimage_evaluation
```

Optional arguments are `--batch-size` and `--workers`.

### Score a saved hard-sample set

Purpose: score saved classifier-guided PNG files with a ResNet critic using five fixed crops and report mean fake probability and real/fake classification rates. It also writes one score per image.

Entry point: `tools/evaluate_hard_samples.py`

```bash
PYTHONPATH=src .venv/bin/python -u tools/evaluate_hard_samples.py \
  --checkpoint outputs/resnet50_critic_probe_guided_s20/best.pt \
  --metrics outputs/sd14_resnet50_guidance_coco20k/metrics.jsonl \
  --images-root outputs/sd14_resnet50_guidance_coco20k \
  --output outputs/resnet50_critic_probe_guided_s20/hard_sample_evaluation
```

Optional arguments are `--batch-size` and `--workers`.

## 4. Generate images with classifier guidance

### Generic classifier-guidance command

Purpose: generate SD1.4 images while optionally steering low-noise denoising steps toward the critic's real class. The generator and critic remain frozen. Each run saves its copied critic, model identity, prompt/seed mapping, PNG files, per-step traces, per-image metrics, preview, and resumable state.

Wrapper: `scripts/sample_classifier_guidance.sh`  
Entry point: `tools/sample_classifier_guidance.py`

```bash
bash scripts/sample_classifier_guidance.sh \
  --config CONFIG_PATH
```

Resume an interrupted run without regenerating finished images:

```bash
bash scripts/sample_classifier_guidance.sh \
  --config CONFIG_PATH \
  --resume
```

Available configurations:

| Configuration | Purpose | Output |
|---|---|---|
| `sd14_classifier_guidance.yaml` | ResNet-50, fixed prompt, four matched seeds, strengths 0/1/5/20 | `outputs/sd14_classifier_guidance/` |
| `sd14_classifier_guidance_100_s20.yaml` | ResNet-50, fixed prompt, 100 images at strength 20 | `outputs/sd14_classifier_guidance_100_s20/` |
| `sd14_dinov3_guidance_100.yaml` | GGGT DINOv3-L/16 + original MLP, fixed prompt, 100 matched seeds at strengths 0/8/15/20 | `outputs/sd14_dinov3_guidance_100/` |
| `sd14_dinov3_guidance_coco100.yaml` | Same DINOv3 critic with 100 COCO captions and matched strengths | `outputs/sd14_dinov3_guidance_coco100/` |
| `sd14_resnet50_guidance_coco20k.yaml` | ResNet-50-guided hard set from 20,000 COCO captions, strength 20 | `outputs/sd14_resnet50_guidance_coco20k/` |
| `sd14_original_coco20k.yaml` | Unguided SD1.4 control set with the same 20,000 captions and seeds, strength 0 | `outputs/sd14_original_coco20k/` |

Examples:

```bash
# Small ResNet strength sweep
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_classifier_guidance.yaml

# 20,000 ResNet-guided hard images
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_resnet50_guidance_coco20k.yaml

# Matching 20,000-image unguided control set
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_original_coco20k.yaml
```

The DINOv3 configurations additionally require the local GGGT repository, DINOv3 backbone, and MLP checkpoint at the paths recorded in their YAML files.

### Rebuild comparison images from an existing guidance run

Purpose: recreate `matched_comparison.jpg` and paginated matched-seed grids from existing PNG files and metrics without running SD1.4 again.

```bash
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_dinov3_guidance_coco100.yaml \
  --visualize-only \
  --seeds-per-page 4
```

Use this on small comparison runs. A 20,000-image run would create thousands of pages.

### Export guidance CSV files and one compact matched grid

Purpose: convert a completed small guidance sweep into `scores.csv`, `fake_probabilities_by_seed.csv`, and `matched_comparison.jpg`.

Entry point: `tools/summarize_guidance.py`

```bash
PYTHONPATH=src .venv/bin/python tools/summarize_guidance.py \
  --output outputs/sd14_dinov3_guidance_coco100 \
  --preview-seeds 4
```

This tool requires a completed run whose `summary.json` contains the explicit seed list; large 20,000-image summaries intentionally omit that list.

## 5. Fine-tune the critic with PROBE-style hard data

### Train a critic with online classifier-guided hard negatives

Purpose: initialize a ResNet-50 from ImageNet weights, then repeatedly generate SD1.4
hard negatives against the current frozen detector snapshot and update that same detector
on a balanced batch of COCO real images and new generated images. The manifest contains
20,000 unique COCO image IDs and one caption per image. Guidance warms up at strength zero,
ramps to 20, and is applied only at low-noise predicted-x0 steps.

Wrapper: `scripts/train_online_guided_critic.sh`  
Entry point: `tools/train_online_guided_critic.py`  
Config: `configs/experiments/resnet50_online_guided_coco20k.yaml`

```bash
bash scripts/train_online_guided_critic.sh
```

Resume from the last completed 64-pair block:

```bash
bash scripts/train_online_guided_critic.sh \
  --resume outputs/resnet50_online_guided_coco20k/latest.pt
```

For a bounded smoke run, add `--max-samples 64` and use a separate output path.

### Train on the saved online-guided set with cosine learning-rate decay

Purpose: initialize a fresh ImageNet-pretrained ResNet-50 and train it once over the
20,000 real/generated pairs retained by the online experiment. It preserves ascending
prompt-index order and the same 64-real/64-fake block construction, augmentation,
BF16 microbatching, and AdamW settings. Unlike the online run, it uses the already saved
fakes and decays the learning rate from `1e-4` to `1e-6` over 313 optimizer steps.

Wrapper: `scripts/train_saved_guided_critic.sh`

Entry point: `tools/train_saved_guided_critic.py`

Config: `configs/experiments/resnet50_saved_guided_cosine_coco20k.yaml`

```bash
bash scripts/train_saved_guided_critic.sh
```

Resume from the last completed pair block:

```bash
bash scripts/train_saved_guided_critic.sh --resume
```

### Random-order ablation for the saved online-guided set

Purpose: repeat the saved-data cosine experiment with one deterministic random
permutation of all 20,000 pair indices. Sampling is without replacement, each pair is
seen exactly once, and every setting other than pair order remains unchanged.

Wrapper: `scripts/train_saved_guided_shuffled.sh`

Config: `configs/experiments/resnet50_saved_guided_cosine_shuffled_coco20k.yaml`

```bash
bash scripts/train_saved_guided_shuffled.sh
```

Resume from the last completed shuffled block:

```bash
bash scripts/train_saved_guided_shuffled.sh --resume
```

### Continue the online-guided critic on shuffled saved pairs

Purpose: initialize from the completed online-guided critic, then fine-tune it once more
on a deterministic random permutation of the same 20,000 retained real/generated pairs.
The optimizer is reset and its learning rate follows cosine decay from `1e-4` to `1e-6`
over 313 steps. The source checkpoint path and SHA-256 identity are recorded with the run.

Wrapper: `scripts/train_online_then_saved_guided_shuffled.sh`

Config: `configs/experiments/resnet50_online_then_saved_guided_cosine_shuffled_coco20k.yaml`

```bash
bash scripts/train_online_then_saved_guided_shuffled.sh
```

Resume from the last completed shuffled block:

```bash
bash scripts/train_online_then_saved_guided_shuffled.sh --resume
```

### Train on original data plus 20,000 guided fakes and paired COCO reals

Purpose: initialize from the selected baseline critic and fine-tune the full ResNet-50. Every update combines a 64-image batch from the original SD1.4 training data and a 64-image batch from the 40,000-record PROBE set, using `0.5 × original BCE + 0.5 × PROBE BCE`.

Wrapper: `scripts/finetune_probe.sh`  
Entry point: `tools/finetune_probe_critic.py`  
Default config: `configs/experiments/resnet50_probe_finetune.yaml`

```bash
bash scripts/finetune_probe.sh
```

Resume from the last completed epoch:

```bash
bash scripts/finetune_probe.sh \
  --resume outputs/resnet50_critic_probe_guided_s20/latest.pt
```

Direct Python equivalent:

```bash
PYTHONPATH=src .venv/bin/python -u tools/finetune_probe_critic.py \
  --config configs/experiments/resnet50_probe_finetune.yaml
```

The run snapshots the source critic and exact manifests in its output directory. Start a separate experiment by copying the YAML and changing `output`; do not overwrite a completed run.

### Run the equal-size unguided-data ablation

Purpose: test whether the guided experiment's improvement comes from detector-targeted hard samples or simply from adding more data. This control replaces the 20,000 strength-20 guided fakes with 20,000 ordinary SD1.4 images generated from identical COCO captions and seeds. The baseline checkpoint, paired COCO real images, original GenImage batches, optimizer, batch sizes, augmentations, validation split, and stopping rule are unchanged.

Config: `configs/experiments/resnet50_unguided_sd14_control.yaml`

```bash
bash scripts/finetune_probe.sh \
  --config configs/experiments/resnet50_unguided_sd14_control.yaml
```

Resume:

```bash
bash scripts/finetune_probe.sh \
  --config configs/experiments/resnet50_unguided_sd14_control.yaml \
  --resume outputs/resnet50_critic_unguided_sd14_control/latest.pt
```

## 6. Train and visualize a learnable SD1.4 soft prompt

### Audit the differentiable path

Purpose: verify that gradients pass through CLIP, all DDIM steps, VAE decoding, perceptual features, and the critic to the 16 soft tokens while every model remains frozen.

Wrapper: `scripts/train_prompt.sh`  
Entry point: `tools/train_prompt.py`  
Default config: `configs/experiments/sd14_soft_prompt.yaml`

```bash
bash scripts/train_prompt.sh --audit-only
```

### Train the soft prompt

Purpose: optimize only the 16×768 soft-token tensor with real-target critic BCE plus perceptual regularization. It writes prompt checkpoints, diagnostic and held-out grids, metrics, and early-stopping state.

```bash
bash scripts/train_prompt.sh \
  --config configs/experiments/sd14_soft_prompt.yaml
```

Resume:

```bash
bash scripts/train_prompt.sh \
  --config configs/experiments/sd14_soft_prompt.yaml \
  --resume outputs/sd14_soft_prompt_bce_perceptual/latest.pt
```

### Render a trained prompt checkpoint

Purpose: generate a same-seed comparison with three rows: prefix only, initialized soft tokens, and trained soft tokens. Images and fake probabilities are written under `visualizations/` beside the checkpoint.

Entry point: `tools/visualize_prompt.py`

```bash
PYTHONPATH=src .venv/bin/python -u tools/visualize_prompt.py \
  --checkpoint outputs/sd14_soft_prompt_bce_perceptual/latest.pt
```

## Complete executable inventory

| File | Role |
|---|---|
| `scripts/train.sh` | Shell wrapper for baseline critic training |
| `scripts/evaluate.sh` | Shell wrapper for one official validation split |
| `scripts/evaluate_genimage.sh` | Shell wrapper for all unseen GenImage generators |
| `scripts/sample_classifier_guidance.sh` | Shell wrapper for guided or unguided SD1.4 sampling |
| `scripts/finetune_probe.sh` | Shell wrapper for PROBE-style critic fine-tuning |
| `scripts/train_prompt.sh` | Shell wrapper for soft-prompt auditing/training |
| `tools/train_critic.py` | Baseline critic trainer |
| `tools/evaluate_critic.py` | Single-split critic evaluator |
| `tools/evaluate_genimage.py` | Multi-generator GenImage evaluator |
| `tools/prepare_coco_prompts.py` | COCO caption-manifest builder |
| `tools/sample_classifier_guidance.py` | Classifier-guided SD1.4 sampler and visualization rebuilder |
| `tools/summarize_guidance.py` | Guidance CSV and compact-grid exporter |
| `tools/finetune_probe_critic.py` | PROBE-style critic trainer |
| `tools/evaluate_hard_samples.py` | Saved hard-image scorer |
| `tools/train_prompt.py` | Differentiable soft-prompt trainer and audit |
| `tools/visualize_prompt.py` | Trained soft-prompt renderer |

For the detailed data transformations, loss definitions, checkpoint contents, and classifier-guidance equations, see `README.md`. Experiment results are collected in `Report.md`.
