# FaceDepth

Face-specialized monocular depth estimation. One photo in, sharp facial relief out, with no depth sensor.

[![weights](https://img.shields.io/badge/weights-🤗%20a--ml%2FFaceDepth-yellow)](https://huggingface.co/a-ml/FaceDepth)
[![paper](https://img.shields.io/badge/paper-paper.md-blue)](paper.md)
[![license](https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey)](#license-and-provenance)

General depth models train on streets, rooms, and furniture, so they flatten a face into a smooth blob. A face holds its information in millimeters of relief, the bridge of the nose against the cheek, the small step from lid to eyeball, and that signal sits far below what a scene-trained model represents. Face relighting, avatar capture, and AR effects need the part those models throw away.

FaceDepth fine-tunes Depth Anything V2-Large on 30,000 CelebA-HQ faces, distilling Apple's boundary-accurate Depth Pro teacher under three losses guided by CelebAMask-HQ face parsing. It resolves the eyelid crease, the nostril rim, the lip contour, and the hairline.

![input, teacher, pretrained, FaceDepth, and difference for six held-out faces](figures/comparison.jpg)

*Six held-out faces. Left to right: input, Depth Pro teacher, pretrained DA2-Large, FaceDepth, and the difference between the last two. Depth is normalized inside the face mask so relief stays visible. Compare columns three and four against column two.*

## Results

Measured on 500 held-out CelebA-HQ faces.

| metric | pretrained DA2-Large | FaceDepth | change |
|---|---|---|---|
| face-region SSI-MAE | 0.02764 | **0.00906** | **−67%** |
| depth-edge F1 vs teacher | 0.717 | **0.882** | **+23%** |
| edge recall vs teacher | 0.745 | **0.873** | +17% |
| edge precision vs teacher | 0.694 | **0.893** | +29% |

Edge metrics are density-matched at 5% of in-face gradient pixels with a 2-pixel tolerance, so a blurry model cannot win by spreading weak gradients across the whole face. Recall and precision both rise, so the student finds the teacher's edges and stops inventing edges the teacher does not have.

![validation error falling from 0.0276 to 0.0091 over 17,000 steps](figures/training_curve.png)

Error drops steeply for the first 2,600 steps, then grinds down for another 14,000 as the model refines fine structure. A machine crash near step 11,200 cost nothing: the run resumed from its last checkpoint and continued below the pre-crash best.

## Why the teacher matters most

The teacher sets the ceiling. No student grows sharper than the labels it learns from, which makes teacher choice the dominant lever in the whole recipe.

![the same face under a general depth teacher and under Depth Pro](figures/teacher_compare.jpg)

*Left to right: input, a general DA2-Base teacher at 518 px, Depth Pro at 1024 px. Depth Pro separates individual hair strands where the general teacher gives a smooth gradient.*

## Download

| file | format | size | notes |
|---|---|---|---|
| `FaceDepth_step15792.pt` | PyTorch | 2.5 GB | original best checkpoint. Use the `ema_model` key. |
| `coreml/FaceDepth_fp32.mlpackage` | Core ML | 1.2 GB | unquantized reference, 5.4 fps |
| `coreml/FaceDepth_fp16.mlpackage` | Core ML | 668 MB | **realtime, 40.5 fps, Neural Engine** |
| `coreml/FaceDepth_int8.mlpackage` | Core ML | 335 MB | 39.4 fps, smallest, GPU |

All from [huggingface.co/a-ml/FaceDepth](https://huggingface.co/a-ml/FaceDepth). Every Core ML export correlates at 1.00000 against the PyTorch reference. Benchmarks use 392×518 input on an Apple-silicon laptop with `ComputeUnit.ALL`, averaged over 20 runs after warmup.

On this hardware int8 buys a 2× size reduction rather than speed, so its advantage is the app bundle. The ranking may differ on iPhone, where the Neural Engine is relatively stronger, and we have not measured that.

## Quick start

```python
import numpy as np, coremltools as ct
from PIL import Image

model = ct.models.MLModel("coreml/FaceDepth_fp16.mlpackage",
                          compute_units=ct.ComputeUnit.ALL)
img = Image.open("face.jpg").convert("RGB").resize((392, 518))
disp = np.asarray(model.predict({"image": img})["depth"]).reshape(518, 392)
# inverse depth: larger = nearer
```

ImageNet normalization is folded into the graph, so pass raw pixels.

**Normalize for display inside the face, not across the frame.** A whole-frame min-max collapses the face's range the moment a distant background enters the shot, which reads as a black or washed-out face. This one detail causes most of the "the model looks broken" reports.

```python
lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
norm = np.clip((disp - lo) / (hi - lo), 0, 1)   # 1 = nearest
```

## Method

**Teacher.** Depth Pro labels 30,000 CelebA-HQ faces at native 1024 px, stored as inverse depth. That pass takes about 18 hours.

**Masks.** The 19 CelebAMask-HQ classes collapse into a head foreground mask, a per-pixel feature weight with eyes and brows highest, and a boundary map of feature edges.

**Losses.**

- **Foreground-restricted SSI**, a 20%-trimmed scale-and-shift-invariant error aligned over the head alone. This spends the model's dynamic range on facial relief instead of the background.
- **Feature-weighted gradient matching** on the depth residual, at four scales. This is the term that produces sharpness.
- **Confidence-weighted regression** with a learned per-pixel weight, so the student discounts pixels the teacher labels unreliably.
- **A boundary term** at parsed feature edges, pushing crisp depth steps to the lid line, lip, nostril, and hairline.

**Training.** Multi-resolution crops at 518, 700, and 910 px. AdamW, head and encoder learning rates 2e-5 and 2e-6, cosine schedule, gradient clipping at 1.0, fp32, gradient checkpointing, EMA 0.999. 17,000 steps, stopped on a validation plateau.

Everything ran on one Apple-silicon laptop. No cluster, no cloud GPU.

Full derivations, equations, and the evaluation protocol live in [paper.md](paper.md).

## An evaluation result worth reading

Our first protocol scored depth edges against every parsed feature edge at a fixed gradient threshold. It ranked the sharp teacher **last** and implied that fine-tuning hurt.

Two flaws produced that. Feature masks mark semantic boundaries, an eyebrow or a color line, which are often depth-flat. And a fixed threshold rewards a diffuse model whose weak gradients drift across those boundaries, which showed up as its precision of 0.155. Density matching against the teacher's own depth edges removes both, and the corrected numbers agree with the error curve and the images.

The flawed protocol was internally consistent and would have supported the opposite conclusion. Section 4 of the paper documents it.

## Limitations

This model optimizes single-image sharpness. The temporal-consistency loss came out of the recipe to get there, so live video flickers more than a temporally trained model would. For offline video, smooth the normalization range across frames, and optionally the depth itself, which trades flicker for ghosting under fast motion.

Distillation caps detail at the teacher. The claim is sharp feature relief and boundaries, not sub-millimeter texture. Per-eyelash depth is beyond what any current monocular teacher resolves.

Training data is CelebA-HQ: centered, well-lit, and limited in pose, occlusion, and demographic diversity relative to in-the-wild use. Performance by demographic group has not been measured. Evaluate before deploying on populations or capture conditions that differ from CelebA-HQ.

Loss-term and resolution ablations have not been run, so the contribution attributed to each individual loss term rests on the design argument rather than measured deltas.

## License and provenance

Released under **CC-BY-NC-4.0**, non-commercial research use.

FaceDepth derives from [Depth Anything V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large), which is CC-BY-NC-4.0. It trains on pseudo-labels from [Apple Depth Pro](https://github.com/apple/ml-depth-pro) and on [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), whose terms restrict use to non-commercial research and education. Honor the upstream terms of all three.

## Citation

```bibtex
@software{facedepth2026,
  title  = {FaceDepth: Face-Specialized Monocular Depth by Distilling a
            Boundary-Accurate Teacher under Segmentation-Guided Losses},
  author = {Lintzeris, Aristides},
  year   = {2026},
  url    = {https://github.com/AristidesAI/FaceDepth}
}
```

Please also cite the work this builds on: Depth Anything V2 ([arXiv:2406.09414](https://arxiv.org/abs/2406.09414)), Depth Pro ([arXiv:2410.02073](https://arxiv.org/abs/2410.02073)), MiDaS ([arXiv:1907.01341](https://arxiv.org/abs/1907.01341)), DPT ([arXiv:2103.13413](https://arxiv.org/abs/2103.13413)), DINOv2 ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193)), and CelebAMask-HQ ([arXiv:1907.11922](https://arxiv.org/abs/1907.11922)). Full reference list in [paper.md](paper.md#7-references).
