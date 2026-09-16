# ResNet-50 critic: phase 1

For a grouped list of every runnable workflow and copy-paste commands, see
[`RUNBOOK.md`](RUNBOOK.md).

Fine-tune a local ImageNet ResNet-50 to output one logit: sigmoid(logit) is
the probability of an AI-generated image. `nature=0`, `ai=1`. Both backbone
and head are trained. Stable Diffusion is never loaded by these workflows.

## Resources and launch

The current configuration uses one CUDA GPU with BF16 support and a Python environment
containing PyTorch, torchvision, Pillow, NumPy, PyYAML, and SwanLab. Scripts use
`.venv/bin/python`; override with the `PYTHON` environment variable if needed.
The default paths and hyperparameters are in
`configs/experiments/resnet50_critic.yaml`. The pretrained path may be a
torchvision ResNet-50 state dictionary or a directory containing exactly one
`.pth`, `.pt`, or `.bin` file. No weights are downloaded automatically.

```bash
bash scripts/train.sh
bash scripts/train.sh --resume outputs/resnet50_critic/latest.pt
bash scripts/evaluate.sh \
  --checkpoint outputs/resnet50_critic/best.pt \
  --data-root /mnt/f/datasets/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4
```

Use `--pretrained models/resnet50-11ad3fa6.pth` to use the weights already
downloaded into this repository, and `--data-root PATH` to override the dataset
location. Training requires `PATH/train/nature` and `PATH/train/ai`, with
162,000 images each. Evaluation alone reads `PATH/val/{nature,ai}`.

## Protocol

Split creation sorts relative image paths per class and samples 8,100 held-out
indices with `random.Random(42).sample`, sharing the RNG across real then fake.
Manifests preserve sorted order within each class. This yields 307,800 training
and 16,200 internal validation images. JSONL manifests and their SHA-256 hashes
are saved under `outputs/resnet50_critic/manifests/`. Existing manifests must
match exactly. Official validation and other generators never select checkpoints.

Augmentation order: random bilinear scale (p=.5, .5–2), JPEG (p=.5, integer
quality 50–100), PIL Gaussian blur (p=.5, radius 0–3), Gaussian pixel noise
(p=.5, sigma 0–55; clipped uint8), color jitter (.2 in all four components),
horizontal flip (p=.5), bilinear rotation (−10° to 10°), symmetric zero padding
when needed, random 224 crop, and ImageNet normalization. Scaled dimensions
are rounded to the nearest integer, minimum one pixel.

Validation applies quality-96 JPEG to fake images only, symmetric zero padding,
then four corner and center crops. It averages five logits before BCE and
classification (`logit >= 0` means fake). Balanced accuracy is the mean of
class accuracies. Distributed validation partitions without duplicate images
and broadcasts rank-0 model buffers before evaluation. BatchNorm remains local
to each GPU during training.

Training uses AdamW (lr=1e-4 constant, weight decay=1e-4, betas=.9/.999,
epsilon=1e-8), BF16 autocast, 64 images per microbatch, two gradient accumulation steps, and 8 workers.
The effective batch is 128; incomplete accumulation groups are dropped.
BatchNorm uses each 64-image microbatch. A single-GPU accumulated run is not
bitwise equivalent to the original two-GPU DDP run. This processes 307,712 images per epoch. Maximum 20 epochs;
improvement must exceed .0001; stop after three non-improving epochs.

`latest.pt` is saved atomically after each completed epoch; `best.pt` records
significant validation improvements. Both include model, optimizer, completed
epoch, best score, patience counter, configuration, labels, manifest hashes,
metrics, and RNG states for every rank. Resume restores the next epoch with
deterministic sampler and worker seeds. Load only trusted experiment checkpoints.
Reproducibility assumes unchanged data, package versions, and GPU setup.

Outputs also include `configuration.yaml`, `environment.json`, `metrics.jsonl`,
and `train.log`. Final evaluation writes `official_val_metrics.json` (override
with `--output` to retain separate results for multiple generators).

## Future probing

Load `checkpoint['model']` into `build_critic()`, call `.eval()` and
`.requires_grad_(False)`, and normalize differentiable generator tensors with
ImageNet statistics. Do not wrap the critic forward in `no_grad`: gradients
must flow to generator inputs. Minimizing the fake logit (or BCE against zero)
encourages images classified as real. Probing and detector improvement are
outside this experiment; `models/modules/` is reserved for them.

## Local checks

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

## SwanLab tracking

The configured private project is `aigi-detection`, experiment
`resnet50-sd14-critic`. Set `SWANLAB_API_KEY` in the environment when launching.
The credential is never placed in configuration or checkpoints. Training logs
loss, throughput, learning rate, epoch summaries, and all validation metrics.
SwanLab also captures console output and hardware monitoring. Run identity is
stored in `outputs/resnet50_critic/swanlab_run.json` and reused on resume.
Set `swanlab: null` in a separate configuration to disable cloud logging.

Unreadable training images are replaced deterministically with the next readable
image of the same class within the training manifest, preserving batch size and
labels. At most 32 candidates are attempted before raising an error. Original
split manifests remain unchanged. Failures are recorded in `corrupt_train.jsonl`.
Validation never substitutes images: unreadable images are excluded, with actual
class counts and `skipped_real`/`skipped_fake` reported in metrics and SwanLab.
Internal validation failures go to `corrupt_validation.jsonl`; official evaluation
uses a `.corrupt.jsonl` file beside its metrics. Failure logs can repeat across
workers/epochs. Only image opening/decoding errors are handled; transformation and
model errors still propagate.

For two-GPU hardware, use a configuration with `world_size: 2` and
`gradient_accumulation_steps: 1`, then launch `tools/train_critic.py` through
`torch.distributed.run --standalone --nproc_per_node=2` with `PYTHONPATH=src`.

## Caption-free SD 1.4 soft-prompt pilot

```bash
bash scripts/train_prompt.sh --audit-only
bash scripts/train_prompt.sh
bash scripts/train_prompt.sh --resume outputs/sd14_soft_prompt_bce_perceptual/latest.pt
```

`configs/experiments/sd14_soft_prompt.yaml` configures the local SD model, selected
full-finetune ResNet-50 critic, and 16 × 768 trainable embeddings after the fixed
prefix `a realistic photo`. CLIP's ordinary embedding forward is used with a
temporary embedding substitution; positional embeddings, attention, EOS, and
padding retain ordinary CLIP behavior. No vocabulary or model weights are trained.

Generation uses BF16 U-Net/VAE, FP32 text encoder, 512 pixels, DDIM 35 steps,
guidance 7.5, and batch one. Every denoising step participates in backpropagation;
whole-U-Net and VAE activation checkpointing recomputes activations during backward.
There is no truncated-gradient mode. Critic scoring uses differentiable tensor
corner/center crops and mean logits. JPEG is deliberately omitted from the
gradient path; these scores are not the JPEG-96 GenImage evaluation protocol.

Training now uses `BCEWithLogits(critic(image), 0) + 10000 * perceptual_distance`.
There is no prompt penalty. Perceptual distance is the mean squared difference of
channel-normalized layer2/layer3 features from an independent frozen ImageNet
ResNet-50, comparing generated images with prefix-only references from identical
noise seeds. Reference generation has no gradients. The perceptual coefficient
is 10000, matching the earlier three-loss run's effective coefficient. Raw BCE,
raw perceptual distance, and its weighted contribution are logged separately.
The current output is `outputs/sd14_soft_prompt_bce_perceptual/`; previous variants
remain preserved in their own output directories.

AdamW uses lr=1e-3, no weight decay, and norm clipping at 1.0. Up to 500 fresh-seed
updates are run. After update 100, 25 consecutive actual prompt updates with RMS
change ≤ `1e-6 + 1e-3 * previous_prompt_RMS` trigger early stopping. The counter is
saved and restored. The stopping criterion measures optimizer parameter movement,
not loss or noisy single-image critic scores.

Fixed diagnostic and held-out seeds are separate from training seeds. Every 50
updates (and at early stopping), comparison grids show prefix-only, initialized,
and learned prompts as three rows with identical seeds in each column. JSON files
record critic scores, feature deviation, pairwise feature diversity, and saturation.
Candidate selection requires held-out diversity ≥80% of prefix diversity,
perceptual deviation ≤2× initialized deviation, and saturation ≤15%, then chooses
the lowest held-out mean fake logit. `best_candidate.pt` requires visual review;
these proxy gates alone do not establish plausible images or improved detection.

The output includes an immutable critic copy and SHA-256 identity, split seed
lists, configuration, full-gradient audit, losses, per-step resumable `latest.pt`,
50-step prompt snapshots, grids, selection records, and completion reason. The
audit verifies frozen model hashes, nonzero gradients, optimizer isolation, and
fixed-seed save/reload reproducibility. If `SWANLAB_API_KEY` is set, training also
logs to the private `aigi-detection` project. Credentials are not saved in outputs.

Larger hard-example generation and critic retraining are deferred until the pilot
passes visual review. That comparison must use equal generated sample counts,
identical seeds and detector-training budgets, and an explicitly chosen real-image
sampling policy; there are no caption-matched real counterparts in this experiment.

## Low-noise classifier guidance

```bash
bash scripts/sample_classifier_guidance.sh
```

This separate inference experiment freezes SD1.4 and the evaluated ResNet-50
checkpoint, keeps the text `a realistic photo` fixed, and guides sampling toward
the critic's real label (0). No prompt or model is trained. Configuration is in
`configs/experiments/sd14_classifier_guidance.yaml`; outputs go to
`outputs/sd14_classifier_guidance/`. The initial sweep uses strengths 0, 1, 5,
and 20 with four shared seeds, 512 pixels, DDIM 35 steps, and CFG 7.5.

Following the forward-guidance idea in [Universal Guidance for Diffusion
Models](https://arxiv.org/abs/2302.07121), at training timestep t ≤ 200 (the last
8 DDIM steps here), estimate clean latent
`z0 = (zt - sqrt(1-alpha_bar_t) * epsilon_CFG) / sqrt(alpha_bar_t)`.
Decode z0 through the VAE and compute `L = BCEWithLogits(D(image), 0)` using
differentiable normalized five-crop scoring. Compute the full local gradient
`g = dL/dzt`, including the U-Net prediction's Jacobian; do not detach epsilon.
The critic never sees the noisy latent as an image. Reference to x0 here means
the one-step denoised estimate, not a separate fully sampled clean reference.

Cap the gradient's per-image RMS at 0.1, then use
`epsilon_guided = epsilon_CFG + strength * sqrt(1-alpha_bar_t) * g_clipped`
in the DDIM step. The plus sign is intentional: increasing epsilon in this
direction makes the reverse update descend the real-target BCE. Each guided step
uses a fresh local graph, with U-Net/VAE activation checkpointing. Guidance
arithmetic uses FP32; model computations and stored sampling latents use BF16.
No classifier guidance is applied at high-noise steps. Strength zero is checked
against ordinary sampling exactly. No JPEG, perceptual loss, or prompt penalty
is applied. Model hashes are checked before and after the experiment.

Saved artifacts include the copied critic and its hash, config, per-step clean
estimate scores/gradient magnitudes, final scores, matched-seed grids, and x0
estimate grids. Lower fake probability is evidence of fooling this critic only;
visual quality and unseen-generator detector improvement require separate checks.

The 100-image strength-20 run uses new seeds 22001–22100:

```bash
bash scripts/sample_classifier_guidance.sh --config configs/experiments/sd14_classifier_guidance_100_s20.yaml
```

Results are saved under `outputs/sd14_classifier_guidance_100_s20/`. In addition
to the floating-point generation scores, each PNG is loaded from disk and scored
again to record whether critic fooling survives 8-bit storage. Individual images,
per-step logs, and 16-image contact sheets are retained. This single-strength run
omits the unguided baseline and reports baseline-difference metrics as null.

The DINOv3 comparison uses the original (unfiltered) head from GGGT, with 100
matched seeds at strengths 0, 8, 15, and 20:

```bash
bash scripts/sample_classifier_guidance.sh --config configs/experiments/sd14_dinov3_guidance_100.yaml
```

Results are under `outputs/sd14_dinov3_guidance_100/`. The adapter loads GGGT's
DINOv3-L/16 implementation, normalizes the CLS embedding, and applies the trained
1024→768→1 GELU MLP. It uses GGGT's FP16 autocast for the critic (SD remains BF16).
Guidance resizes the full RGB estimate to 224×224 with differentiable antialiased
bilinear interpolation, followed by ImageNet normalization. Saved PNG scores use
GGGT's original PIL resize/normalization, so any discrepancy from differentiable
tensor preprocessing is retained in the two score columns. The spectral filter
and filtered MLP head are not used. The head checkpoint is copied into the output
directory; the backbone hash and a snapshot of its implementation are recorded.

The COCO-caption comparison uses the first 100 records from the paper's ordered
20,000-prompt manifest, paired with the same seeds at all four strengths:

```bash
PYTHONPATH=src .venv/bin/python tools/prepare_coco_prompts.py \
  --annotations /path/to/annotations_trainval2014.zip \
  --output outputs/coco2014_prompts/first_20000.jsonl
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_dinov3_guidance_coco100.yaml
.venv/bin/python tools/summarize_guidance.py \
  --output outputs/sd14_dinov3_guidance_coco100
```

This experiment changes only the conditioning text in our inference-time
classifier-guidance sweep. It does not implement PROBE's rank-128 U-Net LoRA or
Deep Reward Tuning optimization. Prompt order, annotation IDs, image IDs, seeds,
per-image scores, and the exact 100-prompt subset are saved with the outputs.

The 20,000-image ResNet-50 guidance run uses every prompt in that ordered manifest
once, with seeds 0–19,999 and guidance strength 20:

```bash
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_resnet50_guidance_coco20k.yaml

# Continue after an interruption without regenerating completed images:
bash scripts/sample_classifier_guidance.sh \
  --config configs/experiments/sd14_resnet50_guidance_coco20k.yaml --resume
```

Large runs encode captions lazily and stream images, metrics, and step traces to
disk. `run_state.json` is updated every 25 images. The first 16 generated images
form a compact preview; no all-image tensor grid is retained in memory.
