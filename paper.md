# FaceDepth: Face-Specialized Monocular Depth by Distilling a Boundary-Accurate Teacher under Segmentation-Guided Losses

**Draft, arXiv preprint style.** Main results are measured and final. The ablation tables are marked as not-yet-run and no numbers are invented for them. Every reference in Section 7 was verified against arXiv, Semantic Scholar, or the publisher's own citation page on 27 July 2026.

Model weights: [huggingface.co/a-ml/FaceDepth](https://huggingface.co/a-ml/FaceDepth) · Code and paper: [github.com/AristidesAI/FaceDepth](https://github.com/AristidesAI/FaceDepth)

---

## Abstract

We train a monocular depth model that resolves the fine relief of a human face, the eyelid crease, the nostril rim, the lip contour, the hairline, from a single photo and no depth sensor. General depth models learn from scenes and rooms, so they flatten faces into a smooth blob and discard the structure a face application depends on. FaceDepth distills a boundary-accurate high-resolution teacher, Depth Pro [6], into a DINOv2/DPT student, Depth Anything V2-Large [5], and supervises it with three losses derived from CelebAMask-HQ face parsing [11]: a foreground-restricted scale-and-shift-invariant term, a feature-weighted gradient term on the depth residual, and a boundary term that ties depth discontinuities to true feature edges. On 500 held-out CelebA-HQ faces the student cuts face-region scale-invariant error by 67% against its pretrained initialization, from 0.02764 to 0.00906. At matched edge density it reproduces the teacher's depth-edge structure with an F1 of 0.882 against the pretrained model's 0.717, a 23% gain, and both recall and precision rise. Teacher labeling, mask processing, and multi-resolution training to 910 pixels all run unattended on one Apple-silicon laptop, and the exported Core ML model runs at 40 fps on the same machine.

---

## 1. Introduction

Monocular depth estimation improved fast, and the strongest general models still treat a face as a near-frontal lump. Their training data is scenes, streets, and interiors, so they optimize for the geometry that dominates those images: walls, floors, the gross layout of a room. A face carries its information in millimeters of relief, the bridge of the nose against the cheek, the small step from lid to eyeball, and that signal sits far below what a scene-trained model represents. Face relighting, avatar capture, and AR effects need the part these models throw away.

Two obstacles block a face-specialized model. No ground-truth depth exists for in-the-wild face photos, so supervision has to come from another model, and the student never grows sharper than that teacher. Second, a face's informative depth spans a small fraction of the scene range, so a loss that weights every pixel the same spends its budget on background and hair and under-fits the features that carry the face. A model scores well by getting the coarse face-versus-background gap right while smoothing every eyelid and lip into mush.

We attack both. For supervision we replace the usual general teacher with Depth Pro [6], built for crisp object boundaries at high native resolution. On face crops it resolves individual hair strands and the relief of the lid and iris where a general teacher produces a smooth gradient. For the loss we read CelebAMask-HQ [11] per-feature segmentation and make the supervision face-aware: we align the depth scale over the head alone, weight the gradient term toward eyes, brows, nose, and lips, and add a term at parsed feature edges.

**Contributions.**

- A distillation recipe for sharp face depth. Swapping a general depth teacher for a boundary-accurate one is the dominant lever, and a DINOv2/DPT student distilled from it inherits per-feature relief.
- Three segmentation-guided depth losses: foreground-restricted scale alignment, a feature-weighted residual gradient term, and a parsed-boundary term. Each targets a distinct failure of scene-trained face depth.
- An evaluation correction. Scoring depth edges against all parsed feature edges at a fixed threshold ranks the sharpest model last. We show why, and give a density-matched protocol that agrees with both the error curve and the qualitative results.
- A single-machine pipeline. It labels 30,000 faces, processes their masks, and trains a ViT-Large student at 910-pixel input on one laptop, with a plateau-based stop and a health watchdog that saves and halts before a degrading machine can crash the run.

## 2. Related Work

**Scale-invariant monocular depth.** Eigen et al. [1] introduced the scale-invariant objective for depth from a single image. MiDaS [2] established scale-and-shift-invariant losses with multi-scale gradient matching and trimmed robustness for zero-shot transfer across datasets, and that loss family is the base we build on. Depth Anything V2 [5] scaled the approach on a DINOv2 [4] encoder with a DPT [3] head and synthetic-label distillation. We take its Large variant as the student initialization.

**Sharp and high-resolution depth.** Miangoleh et al. [7] showed that detail in monocular depth scales with input resolution, through content-adaptive multi-resolution merging. Depth Pro [6] produces sharp metric boundaries at high resolution in one forward pass and introduces boundary-accuracy metrics that motivate our evaluation. Ke et al. [8] reach fine texture by repurposing a diffusion image generator, at higher cost. We run Depth Pro as an offline teacher and trade its latency, which does not matter offline, for its boundary accuracy.

**Uncertainty-weighted regression.** Kendall and Gal [9] derived the aleatoric regression loss that lets a network predict a per-pixel confidence and discount its own unreliable targets. Face Anything [10] applies a confidence-weighted regression term and a residual gradient term to face geometry in a multi-view 4D reconstruction setting. We adopt both terms for a single-image, single-model predictor and drive them with parsed face masks.

**Face geometry and parsing.** Parametric shape models such as FLAME [13] and multi-view capture underlie recent 4D face work [10], and both need fitting infrastructure or calibrated rigs. FaceDepth stays a single feed-forward monocular predictor and avoids that machinery. CelebAMask-HQ [11], built on CelebA [12], supplies the 19-class parsing that we turn into depth-loss weighting.

## 3. Method

### 3.1 Teacher labeling

Depth Pro [6] predicts metric depth for each face at native 1024-pixel resolution in about two seconds on the target laptop. We store inverse depth, $d^{*} = 1/z$, which matches the student's output convention and the scale-and-shift-invariant loss space of MiDaS [2]. We run the teacher at native resolution without patch merging [7]. Depth Pro already resolves per-strand structure at native scale, and patch merging would multiply an 18-hour offline pass for a gain we did not need. Section 6 revisits this.

### 3.2 Mask-aware corpus

For each image we collapse the 19 CelebAMask-HQ [11] class masks into three tensors. The foreground mask $M$ marks the head, the union of skin, features, and hair, and excludes background, cloth, and hat. The feature-weight map $w$ assigns higher weight to eyes and brows, medium to nose and lips, and base weight to skin and hair. The boundary map $B$ holds the morphological edges of the eye, brow, nose, lip, and hair masks, the places where a depth discontinuity should fall.

### 3.3 Loss

Write $p$ for the student disparity and $d^{*}$ for the teacher disparity. We first align $p$ to $d^{*}$ by a per-sample least-squares scale and shift computed over the foreground alone, so the full dynamic range serves facial relief rather than the background:

$$ (s, t) = \arg\min_{s,t} \sum_i M_i (s p_i + t - d^{*}_i)^2, \qquad \hat{p} = s p + t $$

The foreground scale-invariant term is a 20%-trimmed mean absolute error over foreground pixels, following the trimmed robust loss of MiDaS [2]:

$$ \mathcal{L}_{\mathrm{ssi}} = \mathrm{trim}_{0.2} \big( M_i | \hat{p}_i - d^{*}_i | \big) $$

where $\mathrm{trim}_{0.2}$ averages the foreground residuals after discarding the largest 20%.

The gradient term is where sharpness comes from. We match multi-scale gradients on the residual $E = \hat p - d^{*}$ and weight each pixel by $w$, combining the residual formulation of Face Anything [10] with the multi-scale gradient matching of MiDaS [2]:

$$ \mathcal{L}_{\mathrm{grad}} = \frac{1}{S}\sum_{s=1}^{S} \left( \frac{\sum w^{(s)} |\nabla_x E^{(s)}|}{\sum w^{(s)}} + \frac{\sum w^{(s)} |\nabla_y E^{(s)}|}{\sum w^{(s)}} \right), \quad S = 4 $$

A confidence head predicts a positive per-pixel weight $W$ from the DPT penultimate features. Following Kendall and Gal [9] and Face Anything [10], the confidence term lets the student discount pixels the teacher labels unreliably:

$$ \mathcal{L}_{\mathrm{conf}} = \frac{\sum M \left( | \hat{p} - d^{*} | W - \alpha \log W \right)}{\sum M}, \qquad \alpha = 0.2 $$

The boundary term concentrates a residual-gradient penalty at parsed feature edges $B$, pushing crisp depth steps to the lid line, lip, nostril, and hairline:

$$ \mathcal{L}_{\mathrm{bnd}} = \frac{\sum B |\nabla_x E|}{\sum B} + \frac{\sum B |\nabla_y E|}{\sum B} $$

The full objective is $\mathcal{L} = \mathcal{L}_{\mathrm{ssi}} + \lambda_g \mathcal{L}_{\mathrm{grad}} + \lambda_c \mathcal{L}_{\mathrm{conf}} + \lambda_b \mathcal{L}_{\mathrm{bnd}}$ with $\lambda_g = 1.0$, $\lambda_c = 0.2$, $\lambda_b = 0.5$.

### 3.4 Architecture and training

The student is Depth Anything V2-Large [5], a DINOv2-L/14 encoder [4] with a DPT head [3], loaded from public weights. A two-convolution confidence head attaches to the DPT penultimate feature through a forward hook, so the pretrained head loads without modification, and we drop that head at inference. We train on square crops at 518, 700, and 910 pixels, all multiples of the patch size, with 910 as the base [7]. The optimizer is AdamW with head and encoder learning rates of $2\times10^{-5}$ and $2\times10^{-6}$ on a cosine schedule, gradient clipping at 1.0, and gradient checkpointing on the encoder so the 910-pixel activations fit in memory. We train in fp32 because bf16 backward passes go numerically unstable on the target Apple-silicon backend. We export the EMA weights, decay 0.999. Training stops on a plateau: after three validations without a meaningful gain, the learning rate anneals over a short cooldown and the run ends.

## 4. Experiments

**Setup.** We label 30,000 CelebA-HQ faces [11, 12] with Depth Pro and hold out 500 for validation. We measure two quantities. Face-region scale-invariant error tracks relief accuracy over the head. Depth-edge agreement tracks whether the model puts depth discontinuities where the teacher does, and it is the metric that reflects sharpness.

**Face-region error.** The student reached a validation scale-invariant error of 0.00906, down from 0.02764 at its pretrained initialization, a 67% reduction. The curve falls steeply for the first 2,600 steps, from 0.028 to 0.011, then descends slowly for another 14,000 steps as it refines fine structure. A machine crash interrupted training near step 11,200. The run resumed from the last checkpoint and continued below the pre-crash best, which the recovered curve confirms.

**Depth-edge agreement.** We extract depth edges at matched density. For each model and image, the top 5% of in-face gradient pixels count as edges, so a blurry model cannot win by spreading weak gradients across the whole face. We then score each model's edges against the Depth Pro teacher's edges at a 2-pixel tolerance, over 200 validation faces. FaceDepth raises edge F1 from 0.717 to 0.882. Recall and precision both improve, so the student finds the teacher's edges and stops inventing edges the teacher does not have.

**Table 1. Depth-edge agreement with the Depth Pro teacher.** Density-matched at 5%, 2-pixel tolerance, 200 held-out faces.

| model | recall | precision | F1 |
|---|---|---|---|
| pretrained DA2-Large | 0.745 | 0.694 | 0.717 |
| **FaceDepth (ours)** | **0.873** | **0.893** | **0.882** |

**Edge behavior against semantic feature edges.** Table 2 scores each model's depth edges against the parsed feature-mask edges, which mark semantic boundaries rather than depth steps. FaceDepth lands at 0.245, next to the teacher's 0.242, and the pretrained model sits higher at 0.304. The pretrained number is high because it places depth edges at flat semantic boundaries, an eyebrow or a color line with no relief behind it. Matching the teacher, and diverging from the pretrained model, is the target.

**Table 2. Edge agreement with semantic feature-mask edges.** Density-matched, teacher-independent.

| model | recall | precision | F1 |
|---|---|---|---|
| pretrained DA2-Large | 0.321 | 0.292 | 0.304 |
| **FaceDepth (ours)** | 0.248 | 0.247 | 0.245 |
| Depth Pro teacher | 0.248 | 0.241 | 0.242 |

**A correction worth reporting.** Our first protocol scored depth edges against all feature-mask edges at a fixed gradient threshold. It ranked the sharp teacher last, at 0.464 recall against the pretrained model's 0.696, and implied that fine-tuning hurt sharpness. Two flaws produced that result. Feature masks mark semantic boundaries that are often depth-flat, and a fixed threshold rewards a diffuse model whose weak gradients drift across those boundaries, which shows up as its precision of 0.155. Density matching against the teacher's own depth edges removes both flaws, and Table 1 then agrees with the error curve and the qualitative results. We report this because the flawed protocol was self-consistent and would have supported the opposite conclusion.

**Deployment.** We export the student to Core ML at three precisions and benchmark on an Apple-silicon laptop, 392x518 input, correlation against the PyTorch reference of 1.00000 in all three cases.

**Table 3. Core ML export.** ComputeUnit.ALL, mean over 20 runs after warmup.

| precision | size | latency | fps |
|---|---|---|---|
| fp32 | 1335 MB | 186.5 ms | 5.4 |
| **fp16** | 668 MB | **24.7 ms** | **40.5** |
| int8 | 335 MB | 25.4 ms | 39.4 |

The int8 and fp16 models both reach roughly 40 fps, and they land on different compute units. fp16 runs on the Neural Engine, int8 on the GPU, where int8 measures 25.3 ms against fp16's 89.2 ms. On this hardware int8 buys a 2x size reduction rather than speed. Ranking may differ on iPhone, where the Neural Engine is relatively stronger, and we have not measured that.

**Qualitative.** Figure 1 shows six held-out faces as input, teacher, pretrained, FaceDepth, and difference. FaceDepth tracks the teacher across skin tones, hair types, a profile, glasses, and a fringe. On the fringe the pretrained model breaks the hair into bright vertical stripes while FaceDepth renders coherent relief. The difference map concentrates on the face interior and hairline, where the mask-weighted losses steer capacity.

**Ablations, not yet run.** The teacher, loss, and resolution ablations need dedicated training runs and are absent from this draft. We report the qualitative teacher comparison and leave the matched quantitative teacher ablation, the per-term loss ablation, and the resolution sweep as planned experiments.

## 5. Reproducibility

The pipeline runs on one instance (aws sagemaker) with no cluster. Teacher labeling of 30,000 images takes about 18 hours at 0.48 images per second. Training ran 17,000 steps at roughly 3 seconds per step across the three resolutions. 
## 6. Limitations

FaceDepth optimizes still-image sharpness and drops the temporal-consistency objective we use for video, so it flickers more on a live feed than a temporally trained variant. The two are separate checkpoints, not one model. For offline video we stabilize at inference time with an EMA on the normalization range and an optional EMA on depth, which reduces flicker at the cost of ghosting under fast motion. Distillation caps detail at the teacher, so a literal per-eyelash depth stays out of reach and we claim sharp feature relief and boundaries, not sub-millimeter texture. We skip patch-merge teacher labeling [7], which could raise the ceiling at a large offline cost. Supervision is CelebA-HQ-centric, with limited pose, occlusion, and skin-tone diversity relative to in-the-wild use, and we have not measured performance by demographic group. The ablation tables remain to be run, so the causal weight we assign to each loss term rests on the design argument and the qualitative comparison rather than measured deltas.

## 7. References

[1] David Eigen, Christian Puhrsch, Rob Fergus. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." NeurIPS, 2014. arXiv:1406.2283.

[2] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun. "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer." 2019. arXiv:1907.01341.

[3] René Ranftl, Alexey Bochkovskiy, Vladlen Koltun. "Vision Transformers for Dense Prediction." ICCV, 2021. arXiv:2103.13413.

[4] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski. "DINOv2: Learning Robust Visual Features without Supervision." 2023. arXiv:2304.07193.

[5] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao. "Depth Anything V2." 2024. arXiv:2406.09414.

[6] Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun. "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second." 2024. arXiv:2410.02073.

[7] S. Mahdi H. Miangoleh, Sebastian Dille, Long Mai, Sylvain Paris, Yağız Aksoy. "Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging." CVPR, 2021. arXiv:2105.14021.

[8] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, Konrad Schindler. "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation." 2023. arXiv:2312.02145.

[9] Alex Kendall, Yarin Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS, 2017. arXiv:1703.04977.

[10] Umut Kocasari, Simon Giebenhain, Richard Shaw, Matthias Nießner. "Face Anything: 4D Face Reconstruction from Any Image Sequence." 2026. arXiv:2604.19702.

[11] Cheng-Han Lee, Ziwei Liu, Lingyun Wu, Ping Luo. "MaskGAN: Towards Diverse and Interactive Facial Image Manipulation." 2019. arXiv:1907.11922. (CelebAMask-HQ dataset.)

[12] Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang. "Deep Learning Face Attributes in the Wild." ICCV, 2015. (CelebA dataset.)

[13] Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero. "Learning a model of facial shape and expression from 4D scans." ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 36(6):194:1–194:17, 2017. doi:10.1145/3130800.3130813.

### BibTeX

```bibtex
@inproceedings{eigen2014depth,
  title     = {Depth Map Prediction from a Single Image using a Multi-Scale Deep Network},
  author    = {Eigen, David and Puhrsch, Christian and Fergus, Rob},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2014},
  eprint    = {1406.2283},
  archivePrefix = {arXiv}
}

@article{ranftl2019midas,
  title   = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  author  = {Ranftl, Ren\'e and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
  journal = {arXiv preprint arXiv:1907.01341},
  year    = {2019},
  eprint  = {1907.01341},
  archivePrefix = {arXiv}
}

@inproceedings{ranftl2021dpt,
  title     = {Vision Transformers for Dense Prediction},
  author    = {Ranftl, Ren\'e and Bochkovskiy, Alexey and Koltun, Vladlen},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  eprint    = {2103.13413},
  archivePrefix = {arXiv}
}

@article{oquab2023dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and Darcet, Timoth\'ee and Moutakanni, Th\'eo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Assran, Mahmoud and Ballas, Nicolas and Galuba, Wojciech and Howes, Russell and Huang, Po-Yao and Li, Shang-Wen and Misra, Ishan and Rabbat, Michael and Sharma, Vasu and Synnaeve, Gabriel and Xu, Hu and Jegou, Herv\'e and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal = {arXiv preprint arXiv:2304.07193},
  year    = {2023},
  eprint  = {2304.07193},
  archivePrefix = {arXiv}
}

@article{yang2024depthanythingv2,
  title   = {Depth Anything V2},
  author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal = {arXiv preprint arXiv:2406.09414},
  year    = {2024},
  eprint  = {2406.09414},
  archivePrefix = {arXiv}
}

@article{bochkovskii2024depthpro,
  title   = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  author  = {Bochkovskii, Aleksei and Delaunoy, Ama\"el and Germain, Hugo and Santos, Marcel and Zhou, Yichao and Richter, Stephan R. and Koltun, Vladlen},
  journal = {arXiv preprint arXiv:2410.02073},
  year    = {2024},
  eprint  = {2410.02073},
  archivePrefix = {arXiv}
}

@inproceedings{miangoleh2021boosting,
  title     = {Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging},
  author    = {Miangoleh, S. Mahdi H. and Dille, Sebastian and Mai, Long and Paris, Sylvain and Aksoy, Ya\u{g}{\i}z},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  eprint    = {2105.14021},
  archivePrefix = {arXiv}
}

@article{ke2023marigold,
  title   = {Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author  = {Ke, Bingxin and Obukhov, Anton and Huang, Shengyu and Metzger, Nando and Daudt, Rodrigo Caye and Schindler, Konrad},
  journal = {arXiv preprint arXiv:2312.02145},
  year    = {2023},
  eprint  = {2312.02145},
  archivePrefix = {arXiv}
}

@inproceedings{kendall2017uncertainties,
  title     = {What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?},
  author    = {Kendall, Alex and Gal, Yarin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017},
  eprint    = {1703.04977},
  archivePrefix = {arXiv}
}

@article{kocasari2026faceanything,
  title   = {Face Anything: 4D Face Reconstruction from Any Image Sequence},
  author  = {Kocasari, Umut and Giebenhain, Simon and Shaw, Richard and Nie{\ss}ner, Matthias},
  journal = {arXiv preprint arXiv:2604.19702},
  year    = {2026},
  eprint  = {2604.19702},
  archivePrefix = {arXiv}
}

@article{lee2019maskgan,
  title   = {MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author  = {Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  journal = {arXiv preprint arXiv:1907.11922},
  year    = {2019},
  eprint  = {1907.11922},
  archivePrefix = {arXiv}
}

@inproceedings{liu2015celeba,
  title     = {Deep Learning Face Attributes in the Wild},
  author    = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2015}
}

@article{FLAME:SiggraphAsia2017,
  title   = {Learning a model of facial shape and expression from {4D} scans},
  author  = {Li, Tianye and Bolkart, Timo and Black, Michael. J. and Li, Hao and Romero, Javier},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume  = {36},
  number  = {6},
  year    = {2017},
  pages   = {194:1--194:17},
  url     = {https://doi.org/10.1145/3130800.3130813}
}
```

## 8. License and provenance

FaceDepth is a derivative of Depth Anything V2-Large, released under CC-BY-NC-4.0, and it is trained on pseudo-labels from Depth Pro and on CelebAMask-HQ, whose terms restrict use to non-commercial research and education. The released weights therefore carry **CC-BY-NC-4.0** and are for non-commercial research use. Anyone building on this work should honor the upstream terms of Depth Anything V2 [5], Depth Pro [6], and CelebAMask-HQ [11, 12].
