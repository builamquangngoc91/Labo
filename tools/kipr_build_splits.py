import os
import random
import pickle
from pathlib import Path


def collect_slide_dirs(images_root: Path):
    # Each slide is a directory; tiles are inside as .png
    return sorted([p for p in images_root.iterdir() if p.is_dir()])


def assign_slides_to_classes(slide_dirs):
    # EDIT: customize your 3 labels and slide mapping rules here.
    # Example: round-robin assignment to 3 classes.
    classes = ["ClassA", "ClassB", "ClassC"]
    cls_to_slides = {c: [] for c in classes}
    for i, sd in enumerate(slide_dirs):
        cls = classes[i % len(classes)]
        cls_to_slides[cls].append(sd)
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
    data_root = repo_root / "datasets/kipr"
    images_root = data_root / "images"
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    slide_dirs = collect_slide_dirs(images_root)
    classes, cls_to_slides = assign_slides_to_classes(slide_dirs)

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

    # Fallback: if slide-level split yields empty train/val for a class (few slides),
    # perform patch-level split to ensure non-empty splits.
    def ensure_non_empty_for_class(cls_name):
        has_train = len(cls2train.get(cls_name, [])) > 0
        has_val = len(cls2val.get(cls_name, [])) > 0
        has_test = len(cls2test.get(cls_name, [])) > 0
        if has_train and has_val and has_test:
            return
        all_slides = cls_to_slides[cls_name]
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
