import os
import shutil

# Paths
src_dir = 'Project/DD2424-Project/images'
target_dir = 'Project/DD2424-Project/data/train'
annotation_file = 'Project/DD2424-Project/annotations/trainval.txt'  # change all 3 into your dir

os.makedirs(os.path.join(target_dir, 'cat'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'dog'), exist_ok=True)

with open(annotation_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        image_name = parts[0] + '.jpg'
        class_id = int(parts[1])  # 1–37
        label = 'cat' if class_id <= 12 else 'dog'  # Classes 1–12: cats, 13–37: dogs
        src_path = os.path.join(src_dir, image_name)
        dst_path = os.path.join(target_dir, label, image_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
