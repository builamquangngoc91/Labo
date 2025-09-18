## KIPR (WSI tiles) quick guide

Setup wandb to tracking (Follow guildline repo)

Structure:

```
kipr/
  images/               # slide folders with .png tiles
  splits/               # generated train/val/test pickles
  concepts/
    class2concepts.json # class → concepts mapping
```

Generated defaults:

- Classes: `ClassA`, `ClassB`, `ClassC` (round-robin per slide folder).
- Splitting: slide-level when possible; if too few slides per class, fallback to patch-level 70/10/20.
- Image extension: `.png`.

Build splits:

```bash
python tools/kipr_build_splits.py
```

Run linear probe:

```bash
sh linear_probe.sh kipr 1 ViT-L/14
```

Run LaBo (1-shot):

```bash
sh labo_train.sh 1 kipr

sh labo_train.sh 1 kipr --cfg-options bs=64 max_epochs=200

sh labo_train.sh 1 kipr --cfg-options bs=1 max_epochs=1
```

LaBo Testing (KIPR):

```bash
# Generic form
sh labo_test.sh cfg/asso_opt/kipr/kipr_1shot_fac.py EXP/CKPT_PATH.ckpt

# Example: test the most recent checkpoint produced by training
sh labo_test.sh cfg/asso_opt/kipr/kipr_1shot_fac.py "$(ls -t exp/asso_opt/kipr/kipr_1shot_fac/*.ckpt | head -n1)"
```

The test accuracy will be appended to `output/asso_opt/kipr.txt`.

Where to customize:

- `datasets/kipr/concepts/class2concepts.json`: change class names and concept lists.
- `tools/kipr_build_splits.py`:
  - function `assign_slides_to_classes` to map slide folders to your labels.
  - change `classes = ["ClassA", "ClassB", "ClassC"]` to your label names.
  - adjust fallback patch split ratios.
- `cfg/linear_probe/kipr.py`:
  - `img_path`, `img_split_path`, `cls_names`, `img_ext`, `clip_model`, `bs`, `n_shots`.
- `cfg/asso_opt/kipr/kipr_base.py`:
  - `concept_root`, `img_path`, `img_split_path`, `num_cls`, `img_ext`, `clip_model`, `num_concept`.
- `cfg/asso_opt/kipr/kipr_1shot_fac.py`:
  - `n_shots`, `bs`, `lr`, `submodular_weights`.
- can config model to better running and setting cuda py using use gpu variable

Notes:

- Split pickles store basenames without extension (e.g., `TCGA-.../patch_123_456`). Keep `img_ext` consistent.
- To regenerate concept arrays (`concepts_raw.npy`, `cls_names.npy`, `concept2cls.npy`), just run LaBo once; it will auto-generate from the JSON on first run.
