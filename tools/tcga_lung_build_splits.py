import os
import json
import random
import pickle
from pathlib import Path


def collect_slide_dirs(images_root: Path):
    # Each slide is a directory; tiles are inside as .png
    return sorted([p for p in images_root.iterdir() if p.is_dir()])


def infer_classes_from_concepts(concepts_path: Path):
    if concepts_path.is_file():
        try:
            data = json.load(open(concepts_path, "r"))
            if isinstance(data, dict) and len(data) > 0:
                return list(data.keys())
        except Exception:
            pass
    # Fallback to common TCGA-LUNG labels
    return ["LUAD", "LUSC"]


def assign_slides_to_classes(slide_dirs, classes):
    # Assign slides by matching class name substring in slide directory name (case-insensitive)
    cls_to_slides = {c: [] for c in classes}
    unmatched = []
    lower_classes = [c.lower() for c in classes]
    for sd in slide_dirs:
        name = sd.name.lower()
        matched = False
        for c, lc in zip(classes, lower_classes):
            if lc in name:
                cls_to_slides[c].append(sd)
                matched = True
                break
        if not matched:
            unmatched.append(sd)

    # Distribute unmatched slides round-robin to keep classes balanced
    if unmatched and classes:
        for i, sd in enumerate(unmatched):
            c = classes[i % len(classes)]
            cls_to_slides[c].append(sd)

    return classes, cls_to_slides


def split_slides(cls_to_slides, train_ratio=0.7, val_ratio=0.1, seed=0):
    random.seed(seed)
    cls2train, cls2val, cls2test = {}, {}, {}
    for cls, slides in cls_to_slides.items():
        slides = slides.copy()
        random.shuffle(slides)
        n = len(slides)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_slides = slides[:n_train]
        val_slides = slides[n_train : n_train + n_val]
        test_slides = slides[n_train + n_val :]
        cls2train[cls] = train_slides
        cls2val[cls] = val_slides
        cls2test[cls] = test_slides
    return cls2train, cls2val, cls2test


def tiles_from_slides(slide_dirs, images_root: Path):
    # Return basenames without extension relative to images_root
    basenames = []
    for sd in slide_dirs:
        for p in sd.glob("*.png"):
            rel = p.relative_to(images_root).as_posix()
            basenames.append(os.path.splitext(rel)[0])
    return basenames


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "datasets/TCGA-LUNG"
    images_root = data_root / "images"
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    classes = infer_classes_from_concepts(data_root / "concepts/class2concepts.json")

    slide_dirs = collect_slide_dirs(images_root)
    classes, cls_to_slides = assign_slides_to_classes(slide_dirs, classes)

    # Build patch-level lists per split, from slide-level splits
    cls2train_slides, cls2val_slides, cls2test_slides = split_slides(cls_to_slides)

    def build_dict(slide_dict):
        res = {}
        for cls, sdirs in slide_dict.items():
            res[cls] = tiles_from_slides(sdirs, images_root)
        return res

    cls2train = build_dict(cls2train_slides)
    cls2val = build_dict(cls2val_slides)
    cls2test = build_dict(cls2test_slides)

    # Ensure non-empty splits per class by falling back to patch-level split if needed
    def ensure_non_empty_for_class(cls_name):
        has_train = len(cls2train.get(cls_name, [])) > 0
        has_val = len(cls2val.get(cls_name, [])) > 0
        has_test = len(cls2test.get(cls_name, [])) > 0
        if has_train and has_val and has_test:
            return
        all_slides = cls_to_slides.get(cls_name, [])
        all_tiles = tiles_from_slides(all_slides, images_root)
        random.seed(0)
        random.shuffle(all_tiles)
        n = len(all_tiles)
        if n == 0:
            return
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.1))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        cls2train[cls_name] = all_tiles[:n_train]
        cls2val[cls_name] = all_tiles[n_train : n_train + n_val]
        cls2test[cls_name] = all_tiles[n_train + n_val :]

    for c in classes:
        ensure_non_empty_for_class(c)

    with open(splits_dir / "class2images_train.p", "wb") as f:
        pickle.dump(cls2train, f)
    with open(splits_dir / "class2images_val.p", "wb") as f:
        pickle.dump(cls2val, f)
    with open(splits_dir / "class2images_test.p", "wb") as f:
        pickle.dump(cls2test, f)

    print("Done. Splits written to:", splits_dir)
    print(
        {
            k: {c: len(v) for c, v in d.items()}
            for k, d in {"train": cls2train, "val": cls2val, "test": cls2test}.items()
        }
    )


if __name__ == "__main__":
    main()


