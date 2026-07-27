# FaceDepth: Face-Specialized Monocular Depth by Distilling a Boundary-Accurate Teacher under Segmentation-Guided Losses

**Draft, arXiv preprint style.** Main results are final and measured. Ablation tables are marked as not-yet-run. Every citation is a placeholder pending programmatic verification; see the citation note at the end. Do not submit before verifying each reference.

---

## Abstract

We train a monocular depth model that resolves the fine relief of a human face, the eyelid crease, the nostril rim, the lip contour, the hairline, from a single photo and no depth sensor. General depth models learn from scenes and rooms, so they flatten faces into a smooth blob and discard the structure a face application depends on. FaceDepth distills a boundary-accurate high-resolution teacher (Depth Pro) into a compact DINOv2/DPT student (Depth Anything V2-Large), and supervises it with three losses derived from CelebAMask-HQ face parsing: a foreground-restricted scale-and-shift-invariant term, a feature-weighted gradient term on the depth residual, and a boundary term that ties depth discontinuities to true feature edges. On 500 held-out CelebA-HQ faces the student cuts face-region scale-invariant error by 67% over its pretrained initialization, and it reproduces the teacher's depth-edge structure with an F1 of 0.88 against the pretrained model's 0.72, a 23% gain at matched edge density. The whole pipeline, teacher labeling, mask processing, and multi-resolution training to 910 pixels, runs unattended on one Apple-silicon laptop.

---

## 1. Introduction

Monocular depth estimation improved fast, and the strongest general models still treat a face as a near-frontal lump. Their training data is scenes, streets, and interiors, so they optimize for the geometry that dominates those images: walls, floors, the gross layout of a room. A face carries its information in millimeters of relief, the bridge of the nose against the cheek, the small step from lid to eyeball, and that signal sits far below what a scene-trained model bothers to represent. Face relighting, avatar capture, and AR effects need exactly the part these models throw away.

Two obstacles block a face-specialized model. No ground-truth depth exists for in-the-wild face photos, so the supervision has to come from another model, and the student can never grow sharper than that teacher. Second, a face's informative depth spans a small fraction of the scene range, so a loss that weights every pixel the same spends its budget on the background and the hair and under-fits the features that carry the face. A model can score well by getting the coarse face-versus-background gap right while smoothing every eyelid and lip into mush.

We attack both. For supervision we replace the usual general teacher with Depth Pro [dp], built for crisp object boundaries at high native resolution. On face crops it resolves individual hair strands and the relief of the lid and iris where our previous general teacher produced a smooth gradient (Figure 1). For the loss we read CelebAMask-HQ [celebamaskhq] per-feature segmentation and make the supervision face-aware: we align the depth scale over the head alone, weight the gradient term toward eyes, brows, nose, and lips, and add a term at parsed feature edges.

**Contributions.**

- A distillation recipe for sharp face depth. Swapping a general depth teacher for a boundary-accurate one is the dominant lever, and a compact DINOv2/DPT student distilled from it inherits per-feature relief. Section 4 and Figure 1 support this.
- Three segmentation-guided depth losses: foreground-restricted scale alignment, a feature-weighted residual gradient term, and a parsed-boundary term. Each targets a distinct failure of scene-trained face depth.
- A single-GPU pipeline that labels 30,000 faces, processes their masks, and trains a ViT-Large student to 910-pixel input on one laptop, with a plateau-based stop and a health watchdog that saves and halts the run before a degrading machine can crash it.

## 2. Related Work

**Scale-invariant monocular depth.** Eigen et al. [eigen] introduced the scale-invariant objective for depth from a single image. MiDaS [midas] established scale-and-shift-invariant losses with multi-scale gradient matching and trimmed robustness for zero-shot transfer across datasets, and that loss family is the base we build on. Depth Anything V2 [da2] scaled the approach on a DINOv2/DPT backbone [dinov2, dpt] with synthetic-label distillation; we take its Large variant as the student initialization.

**Sharp and high-resolution depth.** Miangoleh et al. [boosting] showed that detail in monocular depth scales with input resolution through content-adaptive patch merging. Depth Pro [dp] produces sharp metric boundaries at high resolution in one forward pass. Marigold [marigold] reaches fine texture through diffusion at higher cost. We run Depth Pro as an offline teacher and trade its latency, which does not matter offline, for its boundary accuracy.

**Uncertainty-weighted regression.** Kendall and Gal [kendallgal] derived the aleatoric regression loss that lets a network predict a per-pixel confidence and discount its own unreliable targets. Face Anything [fa] applied a confidence-weighted regression term and a residual gradient-smoothness term to face geometry in a multi-view 4D reconstruction setting. We adopt both terms for a single-image, single-model predictor and drive them with parsed face masks.

**Face geometry and parsing.** Parametric 3DMM and FLAME fitting [flame] and multi-view capture underlie recent 4D face work [fa], and both need fitting infrastructure or calibrated rigs. FaceDepth stays a single feed-forward monocular predictor and avoids that machinery. CelebAMask-HQ [celebamaskhq] supplies the 19-class parsing that we turn into depth-loss weighting.

## 3. Method

### 3.1 Teacher labeling

Depth Pro [dp] predicts metric depth for each face at native 1024-pixel resolution in about two seconds on the target laptop. We store inverse depth, $d^{*} = 1/z$, which matches the student's output convention and the scale-and-shift-invariant loss space of MiDaS [midas]. We run the teacher at native resolution without patch merging [boosting]: Depth Pro already resolves per-strand structure at native scale, and patch merging would multiply an 18-hour offline pass for a gain we did not need. Section 5 revisits this choice.

### 3.2 Mask-aware corpus

For each image we collapse the 19 CelebAMask-HQ [celebamaskhq] class masks into three tensors. The foreground mask $M$ marks the head, the union of skin, features, and hair, and excludes background, cloth, and hat. The feature-weight map $w$ assigns higher weight to eyes and brows, medium to nose and lips, and base weight to skin and hair. The boundary map $B$ holds the morphological edges of the eye, brow, nose, lip, and hair masks, the places where a depth discontinuity should fall.

### 3.3 Loss

Write $p$ for the student disparity and $d^{*}$ for the teacher disparity. We first align $p$ to $d^{*}$ by a per-sample least-squares scale and shift computed over the foreground alone, so the full dynamic range serves facial relief rather than the background:
$$ (s, t) = \arg\min_{s,t} \sum_i M_i\,(s\,p_i + t - d^{*}_i)^2, \qquad \hat p = s\,p + t. $$

The foreground scale-invariant term is a 20%-trimmed mean absolute error over foreground pixels [midas]:
$$ \mathcal{L}_{\mathrm{ssi}} = \operatorname{trim}_{0.2}\{\, M_i\,|\hat p_i - d^{*}_i| \,\}. $$

The gradient term is where sharpness comes from. We match multi-scale gradients on the residual $E = \hat p - d^{*}$ and weight each pixel by $w$, following the residual formulation of Face Anything [fa] and the multi-scale matching of MiDaS [midas]:
$$ \mathcal{L}_{\mathrm{grad}} = \frac{1}{S}\sum_{s=1}^{S}\Big( \frac{\sum w^{(s)}|\nabla_x E^{(s)}|}{\sum w^{(s)}} + \frac{\sum w^{(s)}|\nabla_y E^{(s)}|}{\sum w^{(s)}} \Big),\quad S=4. $$

A confidence head predicts a positive per-pixel weight $W$ from the DPT penultimate features. Following Kendall and Gal [kendallgal] and Face Anything [fa], the confidence term lets the student discount pixels the teacher labels unreliably:
$$ \mathcal{L}_{\mathrm{conf}} = \frac{\sum M\,(|\hat p - d^{*}|\,W - \alpha \log W)}{\sum M},\qquad \alpha = 0.2. $$

The boundary term concentrates a residual-gradient penalty at parsed feature edges $B$, pushing crisp depth steps to the lid line, lip, nostril, and hairline:
$$ \mathcal{L}_{\mathrm{bnd}} = \frac{\sum B\,|\nabla_x E|}{\sum B} + \frac{\sum B\,|\nabla_y E|}{\sum B}. $$

The full objective is $\mathcal{L} = \mathcal{L}_{\mathrm{ssi}} + \lambda_g \mathcal{L}_{\mathrm{grad}} + \lambda_c \mathcal{L}_{\mathrm{conf}} + \lambda_b \mathcal{L}_{\mathrm{bnd}}$ with $\lambda_g = 1.0$, $\lambda_c = 0.2$, $\lambda_b = 0.5$.

### 3.4 Architecture and training

The student is Depth Anything V2-Large, a DINOv2-L/14 encoder with a DPT head [da2, dinov2, dpt], loaded from public weights. A two-convolution confidence head attaches to the DPT penultimate feature through a forward hook, so the pretrained head loads without modification. We train on square crops at three resolutions, 518, 700, and 910 pixels, all multiples of the patch size, with 910 as the base [boosting]. The optimizer is AdamW with head and encoder learning rates of $2\times10^{-5}$ and $2\times10^{-6}$ on a cosine schedule [fa], gradient clipping at 1.0, and gradient checkpointing on the encoder so the 910-pixel activations fit in memory. We train in fp32 because bf16 backward passes go numerically unstable on the target Apple-silicon backend. We export the EMA weights (decay 0.999). Training stops on a plateau: after three validations without a meaningful gain, the learning rate anneals over a short cooldown and the run ends.

## 4. Experiments

**Setup.** We label 30,000 CelebA-HQ [celebamaskhq] faces with Depth Pro and hold out 500 for validation. We measure two quantities. Face-region scale-invariant error tracks relief accuracy over the head. Depth-edge agreement tracks whether the model puts depth discontinuities where the teacher does, and it is the metric that reflects sharpness.

**Face-region error.** The student reached a validation scale-invariant error of 0.00906, down from 0.02764 at its pretrained initialization, a 67% reduction. The curve falls steeply for the first 2,600 steps, from 0.028 to 0.011, then descends slowly for another 14,000 steps as it refines fine structure. A machine crash interrupted training near step 11,200; the run resumed from the last checkpoint and continued below the pre-crash best without lasting damage, which the recovered curve confirms.

**Depth-edge agreement.** We extract depth edges at matched density: for each model and image, the top 5% of in-face gradient pixels count as edges, so a blurry model cannot win by spreading weak gradients across the whole face. We then score each model's edges against the Depth Pro teacher's edges at a 2-pixel tolerance. Table 1 reports the result over 200 validation faces. FaceDepth raises edge F1 from 0.72 to 0.88, and both recall and precision improve, so it finds the teacher's edges and stops inventing edges the teacher does not have.

**Table 1. Depth-edge agreement with the Depth Pro teacher (density-matched, 2-pixel tolerance, 200 held-out faces).**

| model | recall | precision | F1 |
|---|---|---|---|
| pretrained DA2-Large | 0.745 | 0.694 | 0.717 |
| **FaceDepth (ours)** | **0.873** | **0.893** | **0.882** |

**Edge behavior against semantic feature edges.** Table 2 scores each model's depth edges against the parsed feature-mask edges, which mark semantic boundaries rather than depth steps. FaceDepth lands at 0.245, next to the teacher's 0.242, while the pretrained model sits higher at 0.304. The pretrained number is high because it places depth edges at flat semantic boundaries, an eyebrow or a color line with no real relief. Matching the teacher, and diverging from the pretrained model, is the target here.

**Table 2. Edge agreement with semantic feature-mask edges (density-matched, teacher-independent).**

| model | recall | precision | F1 |
|---|---|---|---|
| pretrained DA2-Large | 0.321 | 0.292 | 0.304 |
| **FaceDepth (ours)** | 0.248 | 0.247 | 0.245 |
| Depth Pro teacher | 0.248 | 0.241 | 0.242 |

**A note on the metric.** Our first attempt scored depth edges against all feature-mask edges at a fixed threshold, and it ranked the sharp teacher last and suggested that fine-tuning hurt. The cause was the confound in Table 2 read naively: feature masks mark semantic boundaries that are often depth-flat, and a fixed threshold rewards a diffuse model whose weak gradients drift across those boundaries. Density matching against the teacher's own depth edges (Table 1) removes both problems and agrees with the error curve and the qualitative results.

**Qualitative.** Figure 1 shows six held-out faces as input, teacher, pretrained, FaceDepth, and difference. FaceDepth tracks the teacher across skin tones, hair types, a profile, glasses, and a fringe. On the fringe the pretrained model breaks the hair into bright vertical stripes while FaceDepth renders coherent relief. The difference map concentrates on the face interior and hairline, where the mask-weighted losses steer capacity.

**Ablations (not yet run).** The teacher, loss, and resolution ablations need dedicated training runs and are not in this draft. We report the qualitative teacher comparison in Figure 1 and leave the matched quantitative teacher ablation, the per-term loss ablation, and the resolution sweep as planned experiments. See Section 5.

## 5. Limitations

FaceDepth optimizes still-image sharpness and drops the temporal-consistency objective we use for video, so it will flicker more on a live feed than a temporally trained variant. The two are separate checkpoints, not one model. Distillation caps detail at the teacher: a literal per-eyelash depth is below what any current monocular teacher resolves, so we claim sharp feature relief and boundaries, not sub-millimeter texture. We skip patch-merge teacher labeling [boosting], which could raise the ceiling at a large offline cost. Supervision is CelebA-HQ-centric, with limited pose, occlusion, and skin-tone diversity relative to in-the-wild use. The ablation tables in Section 4 remain to be run, and until they are, the causal weight we assign to each loss term rests on the design argument and Figure 1 rather than measured deltas.

## 6. Conclusion

Sharp face depth comes from a sharp teacher and a face-aware loss, not from a larger student or more images. Distilling Depth Pro into a Depth Anything V2-Large student under foreground scale alignment, a feature-weighted residual gradient term, a confidence term, and a boundary term cut face-region error by 67% and raised depth-edge agreement with the teacher to 0.88 F1, on a pipeline that runs on one laptop.


| `flame` | Learning a Model of Facial Shape and Expression, FLAME (Li et al.) | SIGGRAPH Asia 2017 | no |

Figure 1 source: `exports/comparison_sharpface.jpg`. Teacher-vs-general comparison: `exports/teacher_compare.jpg`. Training curve: `exports/training_curve_sharpface.png`.
