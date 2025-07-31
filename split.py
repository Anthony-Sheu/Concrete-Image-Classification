import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    categories = ['Positive', 'Negative']

    for category in categories:
        src_folder = os.path.join(source_dir, category)
        all_files = os.listdir(src_folder)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * split_ratio)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        for phase, file_list in zip(['train', 'val'], [train_files, val_files]):
            out_dir = os.path.join(dest_dir, phase, category)
            os.makedirs(out_dir, exist_ok=True)
            for file in file_list:
                shutil.copy(os.path.join(src_folder, file), os.path.join(out_dir, file))

split_dataset(source_dir='.', dest_dir='data', split_ratio=0.8)