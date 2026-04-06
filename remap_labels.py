import os
import glob

base_dir = r"c:\Users\TEJA\PycharmProjects\TumorFinder\dataset_brain tumor_correct"
splits = ["Train", "Valid", "Test"]

remapped_count = 0
for split in splits:
    labels_dir = os.path.join(base_dir, split, "Labels")
    if not os.path.exists(labels_dir):
        continue
    
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for file in txt_files:
        with open(file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Remap class id (YOLO format: class x_center y_center width height)
            class_id = int(parts[0]) - 1
            new_line = f"{class_id} " + " ".join(parts[1:]) + "\n"
            new_lines.append(new_line)
            remapped_count += 1
            
        with open(file, "w") as f:
            f.writelines(new_lines)

print(f"Successfully remapped {remapped_count} boundary boxes across {len(splits)} splits.")
