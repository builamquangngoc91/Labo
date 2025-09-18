steps = 8
n_runs = 1
img_path = "datasets/kipr/images"
data_root = "exp/linear_probe/kipr"
img_split_path = "datasets/kipr/splits"
num_cls = 3
unfreeze_clip = False
paper = True
cls_names = ["ClassA", "ClassB", "ClassC"]
img_ext = ".png"
clip_model = "RN50"  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

lr = 1e-3
bs = 128
n_shots = 1
dataset = "kipr"
