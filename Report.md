# Experiment Report

## ResNet-50 Critic Evaluation on GenImage

The selected epoch-2 baseline and the selected fine-tuned ResNet-50 critics were evaluated on the seen SD 1.4 generator and each unseen generator's official GenImage validation split. Generated images were JPEG-compressed at quality 96; real images were unchanged. Predictions use the mean logit from five fixed 224×224 crops, with fake defined as mean logit ≥ 0.

| Method | SD 1.4 | ADM | BigGAN | Midjourney | VQDM | GLIDE | SD 1.5 | Wukong | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 critic | 99.67% | 51.58% | 49.93% | 70.80% | 54.17% | 58.64% | 99.63% | 94.78% | **72.40%** |
| ResNet-50 critic + unguided-data control | 99.77% | 51.98% | 49.96% | 71.81% | 54.32% | 59.65% | 99.70% | 97.10% | **73.04%** |
| ResNet-50 critic + PROBE fine-tuning | 99.73% | 52.10% | 49.92% | 72.33% | 54.80% | 60.33% | 99.69% | 96.51% | **73.18%** |
| ResNet-50 critic + online unguided training | 95.25% | 56.45% | 47.88% | 75.86% | 62.61% | 65.09% | 95.32% | 90.94% | **73.68%** |
| ResNet-50 critic + online guided training | 95.72% | 57.11% | 47.67% | 74.97% | 63.57% | 66.19% | 95.71% | 92.82% | **74.22%** |
| ResNet-50 critic + saved guided data + cosine LR | 92.51% | 60.33% | 43.77% | 75.71% | 64.62% | 73.38% | 92.26% | 91.15% | **74.22%** |
| ResNet-50 critic + shuffled saved guided data + cosine LR | 94.80% | 57.71% | 46.90% | 74.88% | 62.60% | 68.61% | 94.46% | 92.43% | **74.05%** |
| Online-guided critic + shuffled saved-data fine-tuning | 97.18% | 57.79% | 49.01% | 72.63% | 60.89% | 66.85% | 96.98% | 93.75% | **74.38%** |

Adding the unguided SD1.4 control set improved average balanced accuracy by 0.64 percentage points across all eight generators and raised the unseen-generator macro average from 68.50% to 69.22%. The guided hard-sample model reached 73.18% overall and 69.38% on unseen generators, only 0.14 and 0.17 percentage points above the equal-size control. In this single run, most of the improvement therefore comes from adding matched training data, while detector-guided sampling provides a smaller incremental gain.

The online-guided model was initialized from ImageNet ResNet-50 weights and trained only on 20,000 unique COCO real images paired with hard SD1.4 samples generated against the evolving detector. It reached 74.22% across all eight generators and 71.15% across the seven unseen generators. Relative to the baseline critic, this is a gain of 1.82 and 2.65 percentage points, respectively. The improvement is concentrated on ADM, Midjourney, VQDM, and GLIDE; SD-family accuracy decreased slightly, and BigGAN fake accuracy remained effectively zero. This indicates improved transfer to several diffusion-based domains but not a universal fake-image representation.

The matched online-unguided ablation used the same 20,000 unique COCO image-caption pairs, seeds, SD1.4 sampling settings, sequential 64-pair updates, and detector optimization, but kept classifier-guidance strength at zero throughout generation. It reached 73.68% across all eight generators and 70.59% across the seven unseen generators. Online classifier guidance therefore improved the all-generator and unseen-generator averages by 0.54 and 0.56 percentage points, respectively. The guided model improved most on Wukong, GLIDE, and VQDM, while the unguided model was slightly better on Midjourney and BigGAN. This controlled result indicates a small but measurable benefit from adapting generated samples to the evolving detector.

Training a fresh ImageNet ResNet-50 on the retained online-generated set in the same sequential block order, while decaying the learning rate from `1e-4` to `1e-6`, produced 74.215% across all generators and 71.602% across unseen generators. Its overall balanced accuracy is effectively identical to online training (74.218%), with a 0.455-point unseen-generator gain. The cosine model predicts fake more often: macro fake accuracy increased from 52.98% to 60.48%, while macro real accuracy decreased from 95.46% to 87.95%. The result is therefore a threshold tradeoff rather than an unambiguous overall improvement.

Randomly shuffling the same 20,000 retained training pairs before cosine-LR fine-tuning produced 74.048% across all generators and 71.083% across unseen generators. This is 0.168 and 0.519 percentage points below the sequential replay run, respectively. Shuffling shifted the operating point toward real predictions: macro real accuracy rose from 87.95% to 93.24%, while macro fake accuracy fell from 60.48% to 54.85%. The random order therefore did not improve balanced accuracy in this run.

Starting from the online-guided critic and fine-tuning it again on the shuffled retained pairs produced 74.384% across all generators and 71.129% across unseen generators. Relative to its online-guided starting checkpoint, overall balanced accuracy increased by 0.166 percentage points while unseen accuracy decreased by 0.019 points. Macro real accuracy rose from 95.46% to 97.80%, and macro fake accuracy fell from 52.98% to 50.97%, so the extra pass mainly shifted the operating point toward real predictions rather than improving unseen-generator separation.

- [Machine-readable evaluation table](outputs/resnet50_critic/genimage_evaluation/summary.csv)
- [Official SD 1.4 evaluation](outputs/resnet50_critic/genimage_evaluation/stable_diffusion_v_1_4.json)
- [Complete evaluation metadata and results](outputs/resnet50_critic/genimage_evaluation/summary.json)
- [Fine-tuned critic combined results](outputs/resnet50_critic_probe_guided_s20/genimage_evaluation/combined_summary.json)
- [Fine-tuned critic per-generator results](outputs/resnet50_critic_probe_guided_s20/genimage_evaluation/summary.json)
- [Unguided-data control combined results](outputs/resnet50_critic_unguided_sd14_control/genimage_evaluation/combined_summary.json)
- [Unguided-data control per-generator results](outputs/resnet50_critic_unguided_sd14_control/genimage_evaluation/summary.json)
- [Online-guided critic combined results](outputs/resnet50_online_guided_coco20k/genimage_evaluation/combined_summary.json)
- [Online-guided critic per-generator results](outputs/resnet50_online_guided_coco20k/genimage_evaluation/summary.json)
- [Online-unguided critic SD 1.4 result](outputs/resnet50_online_unguided_coco20k/genimage_evaluation/stable_diffusion_v_1_4.json)
- [Online-unguided critic unseen-generator results](outputs/resnet50_online_unguided_coco20k/genimage_evaluation/summary.json)
- [Saved-data cosine critic combined results](outputs/resnet50_saved_guided_cosine_coco20k/genimage_evaluation/combined_summary.json)
- [Saved-data cosine critic per-generator results](outputs/resnet50_saved_guided_cosine_coco20k/genimage_evaluation/summary.json)
- [Shuffled saved-data cosine critic combined results](outputs/resnet50_saved_guided_cosine_shuffled_coco20k/genimage_evaluation/combined_summary.json)
- [Shuffled saved-data cosine critic per-generator results](outputs/resnet50_saved_guided_cosine_shuffled_coco20k/genimage_evaluation/summary.json)
- [Online-then-shuffled critic combined results](outputs/resnet50_online_then_saved_guided_cosine_shuffled_coco20k/genimage_evaluation/combined_summary.json)
- [Online-then-shuffled critic per-generator results](outputs/resnet50_online_then_saved_guided_cosine_shuffled_coco20k/genimage_evaluation/summary.json)

## ResNet-50 Classifier-Guidance Results

We generated four matched examples with the fixed prompt “a realistic photo” and the frozen ResNet-50 critic. Fake probabilities are averaged over seeds 21001–21004.

| Guidance strength | Mean fake probability |
|---:|---:|
| 0 | 98.83% |
| 1 | 76.71% |
| 5 | 15.78% |
| 20 | **5.06%** |

- [Matched-seed comparison](outputs/sd14_classifier_guidance/matched_comparison.jpg)
- [Complete experiment summary](outputs/sd14_classifier_guidance/summary.json)

We then generated 20,000 images with the first 20,000 ordered COCO 2014 training captions, using ResNet-50 guidance at strength 20. The probability below was recomputed from the saved PNG files.

| Scoring critic | Images | Guidance strength | Mean fake probability | Classified as fake | Classified as real |
|---|---:|---:|---:|---:|---:|
| Original ResNet-50 | 20,000 | 20 | 20.74% | 1,466/20,000 (7.33%) | 18,534/20,000 (92.67%) |
| Unguided-data control ResNet-50 | 20,000 | 20 | 54.51% | 11,412/20,000 (57.06%) | 8,588/20,000 (42.94%) |
| PROBE-fine-tuned ResNet-50 | 20,000 | 20 | **69.72%** | 14,793/20,000 (73.97%) | 5,207/20,000 (26.04%) |

Ordinary-data fine-tuning substantially increased sensitivity to the guided samples, but training directly on those hard samples added a further 15.21 percentage points in mean fake probability and 16.91 percentage points in detection rate. Since the PROBE-fine-tuned critic was trained on these exact 20,000 images, this result measures training-set adaptation rather than generalization to new hard samples.

- [20,000-image experiment summary](outputs/sd14_resnet50_guidance_coco20k/summary.json)
- [Per-image results](outputs/sd14_resnet50_guidance_coco20k/metrics.jsonl)
- [Fine-tuned critic hard-sample evaluation](outputs/resnet50_critic_probe_guided_s20/hard_sample_evaluation/summary.json)
- [Unguided-data control hard-sample evaluation](outputs/resnet50_critic_unguided_sd14_control/hard_sample_evaluation/summary.json)

## DINOv3 Classifier-Guidance Results

We generated 100 images for each classifier-guidance strength using the same seeds (22001–22100). The critic was the frozen, original DINOv3 ViT-L/16 backbone with its SD 1.4-trained MLP head. Fake probabilities below were measured from the saved PNG files using the critic's original preprocessing.

| Guidance strength | Mean fake probability | Classified as real |
|---:|---:|---:|
| 0 | 93.17% | 6/100 |
| 8 | 2.98% | 100/100 |
| 15 | 1.16% | 100/100 |
| 20 | **0.84%** | 100/100 |

Guidance was applied only during the final eight low-noise denoising steps. Strength 8 was sufficient to make all 100 generated images cross the critic's real/fake decision boundary. Higher strengths reduced the average fake probability further. Visual inspection still found malformed faces, garbled text, and collage-like scenes, so these results demonstrate critic evasion rather than photographic realism.

- [Matched-seed image comparison](outputs/sd14_dinov3_guidance_100/matched_comparison.jpg)
- [Per-image fake probabilities](outputs/sd14_dinov3_guidance_100/fake_probabilities_by_seed.csv)
- [Complete experiment summary](outputs/sd14_dinov3_guidance_100/summary.json)

## DINOv3 Guidance with COCO 2014 Captions

We repeated the experiment using the first 100 records from the ordered COCO 2014 training-caption annotations. Each caption was paired with one of the same 100 noise seeds, and each caption–seed pair was held fixed across guidance strengths.

| Guidance strength | Mean fake probability | Classified as real |
|---:|---:|---:|
| 0 | 97.46% | 1/100 |
| 8 | 21.19% | 84/100 |
| 15 | 12.80% | 92/100 |
| 20 | **7.58%** | 97/100 |

These saved-PNG scores use the critic's original preprocessing. Compared with the fixed “a realistic photo” prompt, varied COCO captions made detector evasion substantially harder, especially at strength 8. The complete 20,000-caption manifest follows the paper's stored annotation order and retains caption annotation IDs and source image IDs.

- [COCO matched-seed image comparison](outputs/sd14_dinov3_guidance_coco100/matched_comparison.jpg)
- [COCO per-image fake probabilities](outputs/sd14_dinov3_guidance_coco100/fake_probabilities_by_seed.csv)
- [Ordered 20,000-caption manifest](outputs/coco2014_prompts/first_20000.jsonl)
- [Complete COCO experiment summary](outputs/sd14_dinov3_guidance_coco100/summary.json)
