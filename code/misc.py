import cv2

from Evaluator import convert_path
import numpy as np
import os
import random, json
import shutil, math
from collections import Counter


import binarization.img_proc as img_proc
from binarization.img_proc import save_mask_png, load_img

def runU():
    pass


def runB():
    mask = img_proc.binarize_img(
        load_img("C:/Repo/Metrology/data/misc/misc_img.tif"))
    save_mask_png(mask, "C:/Repo/Metrology/data/misc/mask_binarized2.png")



def r():
    from Evaluator import convert_path, Score_stats, print_eval_scores, quality_score
    from pathlib import Path

    labeling_dir = convert_path(r"C:\Repo\Metrology\Images\2.U-Net 6\2.U-Net 6\UNET-6")
    labeling_imgs = sorted(Path(labeling_dir).rglob("*.tif"))
    labeling_imgs = set([ip.name for ip in labeling_imgs])

    double_dir = convert_path(r"C:\Repo\Metrology\Images\3.double U-Net\3.double U-Net\UNET-6-arg")
    double_imgs = sorted(Path(double_dir).rglob("*.tif"))
    double_imgs = set([ip.name for ip in double_imgs])

    print(f"Labeling images: {len(labeling_imgs)}")
    print(f"Double U-Net images: {len(double_imgs)}")

    print("Intersection Set:", len(set(double_imgs) & set(labeling_imgs)))
    print("Difference Set 1:", len(set(double_imgs) - set(labeling_imgs)))
    print("Difference Set 2:", len(set(labeling_imgs) - set(double_imgs)))

    print("double in labeling:", len([ip for ip in double_imgs if ip in labeling_imgs]))
    print("double not in labeling:", len([ip for ip in double_imgs if ip not in labeling_imgs]))

    print("labeling in double:", len([ip for ip in labeling_imgs if ip in double_imgs]))
    print("labeling not in double:", len([ip for ip in labeling_imgs if ip not in double_imgs]))

    print([ip for ip in labeling_imgs if ip not in double_imgs])

def e():
    from Evaluator import convert_path, Score_stats, print_eval_scores, quality_score
    import json

    scores = load_json(convert_path(r"C:\Repo\Metrology\data\JSON\Classical_1203_mask.json"))
    print(max(scores.keys(), key= lambda x: scores[x]), min(scores.keys(), key= lambda x: scores[x]))
    print(Score_stats(scores, title="Classical_1203_mask  Distribution") )

    #TODO CROSS reference with manual curation
    # Evaluator works
    # Train logistical regression on scores

    
    pretty_json = json.load(convert_path(), indent=4)
    print(pretty_json)

def load_json(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return data


def compare_folders(p1, p2):
    """
    Compares two folders and prints the number of files in each,
    and the number of files that share the same name between them.

    Args:
        p1 (str): The file path to the first folder.
        p2 (str): The file path to the second folder.
    """
    print(f"Comparing folders:\n  Folder 1: {p1}\n  Folder 2: {p2}\n")

    try:
        # Get all entries in each directory
        entries1 = os.listdir(p1)
        entries2 = os.listdir(p2)

        # Filter for files only, ignoring subdirectories
        # os.path.isfile() needs the full path
        files1 = {f for f in entries1 if os.path.isfile(os.path.join(p1, f))}
        files2 = {f for f in entries2 if os.path.isfile(os.path.join(p2, f))}

        # --- Calculations ---
        
        # 1. Number of files in each folder
        count1 = len(files1)
        count2 = len(files2)

        # 2. Files in p1 that are also in p2 (intersection)
        common_files = files1.intersection(files2)
        common_count = len(common_files)

        # 3. Files in p2 that are also in p1 (same as above)
        # The intersection is symmetrical, so common_count is the answer for both.
        
        

        if common_count > 0:
            print("\nCommon file names:")
            for f_name in sorted(list(common_files)):
                print(f"  - {f_name}")

    # --- Printing Results ---
        print("--- Folder Statistics ---")
        print(f"Folder 1 ({os.path.basename(p1)}): {count1} files")
        print(f"Folder 2 ({os.path.basename(p2)}): {count2} files")
        
        print("\n--- Cross-Folder Comparison ---")
        print(f"Files in '{os.path.basename(p1)}' also in '{os.path.basename(p2)}': {common_count}")
        print(f"Files in '{os.path.basename(p2)}' also in '{os.path.basename(p1)}': {common_count}")   

    except FileNotFoundError as e:
        print(f"Error: Folder not found.")
        print(f"  Details: {e}")
    except NotADirectoryError as e:
        print(f"Error: One of the paths is not a directory.")
        print(f"  Details: {e}")
    except PermissionError as e:
        print(f"Error: Permission denied.")
        print(f"  Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



import os
import shutil
import sys

c1 = 0
c2 = 0
s = set()
DRY_RUN = False
def process_files_recursively(root_path, action_func):
    """
    Walks through a directory and applies 'action_func' to every file found.
    
    Args:
        root_path (str): The folder to search.
        action_func (callable): A function that accepts a single argument (the full file path).
    """
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' not found.")
        return

    print(f"--- Processing files in '{root_path}' ---\n")

    count = 0
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            
            # Run the passed-in function on this file
            action_func(full_path)
            
            count += 1
            
    print(f"\n--- Finished. Processed {count} files. ---")

def act_func(file_name):
    global c1
    global c2
    global s
    name, ext = file_name.split(".")
    fst, last = name.split("-")
    if ext == "jpg":

        print(file_name)
        
def delete_if_ends_with_01MS(file_path):
    """
    Checks if a filename ends with '01MS' (ignoring extension).
    If yes, it deletes the file.
    
    Matches: 'S04_M1631-01MS.jpg'
    Ignores: 'S04_M1632-02MS.tif'
    """
    filename = os.path.basename(file_path)
    name_stem, extension = os.path.splitext(filename)
    
    # Check if the name part ends with 01MS
    if name_stem.endswith("01MS"):
        if DRY_RUN:
            print(f"[DRY RUN] Would delete: {filename}")
        else:
            try:
                os.remove(file_path)
                print(f"[DELETED] {filename}")
            except OSError as e:
                print(f"[ERROR] Could not delete {filename}: {e}")


def count_files_in_subfolders(root_path):
    """
    Iterates through the immediate subfolders of root_path 
    and counts the total files inside each one recursively.
    """
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' not found.")
        return

    print(f"--- Counting files per subfolder in: '{root_path}' ---\n")

    # Get only the immediate directories in the root path
    try:
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
    except OSError as e:
        print(f"Error reading directory: {e}")
        return

    total_all = 0
    
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        folder_count = 0
        
        # Walk recursively inside this specific subfolder to count files
        for r, d, f in os.walk(folder):
            folder_count += len(f)
            
        print(f"Folder: {folder_name:<20} | Files: {folder_count}")
        total_all += folder_count
        
    print(f"\nTotal files across all folders: {total_all}")    


def move_random_subset(d1_root, d2_root, ratio, dry_run=True):
    """
    Moves a random percentage of files from subfolders in d1_root to 
    matching subfolders in d2_root.

    Args:
        d1_root (str): Source root directory (D1).
        d2_root (str): Destination root directory (D2).
        ratio (float): Percentage of files to move (e.g., 0.1 for 10%).
        dry_run (bool): If True, only prints actions without moving files.
    """
    if not os.path.exists(d1_root):
        print(f"Error: Source path '{d1_root}' not found.")
        return
    if not os.path.exists(d2_root):
        print(f"Error: Destination path '{d2_root}' not found.")
        return

    print(f"--- Starting Transfer (Ratio: {ratio*100}%) ---")
    print(f"Source (D1):      {d1_root}")
    print(f"Destination (D2): {d2_root}")
    print(f"Mode:             {'DRY RUN (Safe)' if dry_run else 'LIVE EXECUTION'}\n")

    # Get immediate subfolders in D1
    try:
        d1_subs = [d for d in os.listdir(d1_root) if os.path.isdir(os.path.join(d1_root, d))]
    except OSError as e:
        print(f"Error reading source directory: {e}")
        return

    total_moved = 0

    for folder_name in d1_subs:
        src_folder = os.path.join(d1_root, folder_name)
        dst_folder = os.path.join(d2_root, folder_name)

        # Check if corresponding folder exists in D2
        if os.path.isdir(dst_folder):
            # Get all files in the source subfolder
            all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
            
            # Calculate how many to move
            count_total = len(all_files)
            count_to_move = int(count_total * ratio)

            if count_to_move > 0:
                # Randomly select files
                files_to_move = random.sample(all_files, count_to_move)
                
                print(f"Processing '{folder_name}': Found {count_total} files. Moving {count_to_move}...")

                for file_name in files_to_move:
                    src_file = os.path.join(src_folder, file_name)
                    dst_file = os.path.join(dst_folder, file_name)

                    if dry_run:
                        print(f"   [DRY RUN] Would move: {file_name}")
                    else:
                        try:
                            shutil.move(src_file, dst_file)
                            # print(f"   [MOVED] {file_name}") # Uncomment for verbose logs
                        except OSError as e:
                            print(f"   [ERROR] moving {file_name}: {e}")
                
                total_moved += count_to_move
            else:
                print(f"Skipping '{folder_name}': Not enough files ({count_total}) to meet ratio.")
        else:
            print(f"Skipping '{folder_name}': No matching folder in destination.")

    print("\n" + "="*30)
    if dry_run:
        print(f"SIMULATION COMPLETE. Would have moved {total_moved} files.")
        print("To execute, change the 'dry_run' argument to False.")
    else:
        print(f"OPERATION COMPLETE. Moved {total_moved} files.")



def generate_structure_map(root_path, output_json_path="structure_map.json"):
    """
    Scans immediate subfolders of 'root_path'. Each subfolder is treated as a 'structure'.
    Finds all files recursively within that structure and maps them in a JSON.
    
    Format:
    {
        "filename1.jpg": "Structure_A",
        "filename2.jpg": "Structure_B"
    }
    """
    if not os.path.exists(root_path):
        print(f"Error: The path '{root_path}' does not exist.")
        return

    structure_map = {}
    duplicate_count = 0
    
    # Get immediate subfolders (which represent the 'Structures')
    try:
        subfolders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    except OSError as e:
        print(f"Error reading directory: {e}")
        return

    print(f"--- Scanning Root: {root_path} ---")
    print(f"Found {len(subfolders)} structure folders: {subfolders}\n")

    for structure_name in subfolders:
        structure_path = os.path.join(root_path, structure_name)
        
        # Walk through this specific structure folder recursively
        for root, dirs, files in os.walk(structure_path):
            for filename in files:
                
                # Check for duplicates (files with same name in different structures)
                if filename in structure_map:
                    print(f"[WARNING] Duplicate filename '{filename}' found.")
                    print(f"    - Existing mapping: {structure_map[filename]}")
                    print(f"    - New found in:     {structure_name} (Ignored)")
                    duplicate_count += 1
                    continue

                # Map the file to its top-level structure folder
                structure_map[filename] = structure_name

    # Write dictionary to JSON file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(structure_map, f, indent=4)
        print(f"\nSuccess! Mapped {len(structure_map)} files.")
        if duplicate_count > 0:
            print(f"Skipped {duplicate_count} duplicate filenames.")
        print(f"JSON saved to: {os.path.abspath(output_json_path)}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")

def validate_curation_status(curated_root, target_root, output_json_path="curation_status.json"):
    """
    Generates labels for a ML dataset in 'target_root'.
    
    Logic:
    1. Iterates through each structure folder in 'target_root'.
    2. Checks if 'curated_root/{Structure}/image' exists.
    3. If YES:
        - Loads all filenames from the 'target' folder.
        - Checks if each file exists in the 'curated/{Structure}/image' folder.
        - Label 1 (Good) if in curated, 0 (Bad) otherwise.
    4. If NO (Curated folder or image subfolder missing):
        - Loads all filenames from the 'target' folder.
        - Label 0 (Bad) for all files.
        
    Output JSON:
    {
        "StructureName": { 
            "file1.tif": 1, 
            "file2.tif": 0 
        }
    }
    """
    if not os.path.exists(curated_root):
        print(f"Error: Curated path '{curated_root}' does not exist.")
        return
    if not os.path.exists(target_root):
        print(f"Error: Target path '{target_root}' does not exist.")
        return

    validation_map = {}
    
    # Get list of structure folders in the TARGET directory
    try:
        target_structures = [d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d))]
    except OSError as e:
        print(f"Error reading Target directory: {e}")
        return

    print(f"--- Starting Dataset Labeling ---")
    print(f"Dataset (Target):    {target_root}")
    print(f"Reference (Curated): {curated_root} (checking inside '/image' subfolders)\n")

    total_files = 0
    total_curated = 0

    for struct_name in target_structures:
        target_struct_path = os.path.join(target_root, struct_name)
        
        # MODIFIED: We now construct the path to the 'image' subfolder
        # Example: C:\...\Curated\ETE\image
        curated_struct_image_path = os.path.join(curated_root, struct_name, "image")
        
        # Initialize output dict for this structure
        validation_map[struct_name] = {}
        
        # Step 1: Check if the 'image' subfolder exists in Curated
        curated_exists = os.path.isdir(curated_struct_image_path)
        
        # Step 2: Load target files (The dataset we want to label)
        try:
            target_files = [f for f in os.listdir(target_struct_path) if os.path.isfile(os.path.join(target_struct_path, f))]
        except OSError:
            print(f"Skipping empty or inaccessible target folder: {struct_name}")
            continue 

        if not curated_exists:
            print(f"Structure '{struct_name}': Missing '{struct_name}/image' in Curated. All {len(target_files)} files labeled 0 (Bad).")
            for filename in target_files:
                validation_map[struct_name][filename] = 0
                total_files += 1
            continue

        # Step 3: If Curated exists, load its files for lookup
        try:
            curated_files_set = set(os.listdir(curated_struct_image_path))
        except OSError:
            print(f"Error reading curated folder contents for '{struct_name}/image'. Treating as empty.")
            curated_files_set = set()

        # Step 4: Check Target files against Curated set
        count_good = 0
        
        for filename in target_files:
            # LOGIC: Good (1) if in curated, Bad (0) otherwise
            if filename in curated_files_set:
                validation_map[struct_name][filename] = 1
                count_good += 1
                total_curated += 1
            else:
                validation_map[struct_name][filename] = 0
            
            total_files += 1

        print(f"Structure '{struct_name}': {count_good}/{len(target_files)} labeled as Good (1).")

    # Save to JSON
    try:
        with open(output_json_path, 'w') as f:
            json.dump(validation_map, f, indent=4)
        print(f"\nLabeling Complete.")
        print(f"Total Files Processed: {total_files}")
        print(f"Total 'Good' Labels:   {total_curated}")
        print(f"JSON saved to: {os.path.abspath(output_json_path)}")
    except IOError as e:
        print(f"Error writing JSON: {e}")



def build_evaluator_dataset(curated_root, training_root, bad_masks_root, output_dir):
    """
    Constructs the file structure for the Evaluator Model.
    
    Logic:
    1. Iterate through ALL images in 'training_root'.
    2. Check if that image exists in 'curated_root'.
    3. If YES (Good):
       - Copy Image from TRAINING -> 1_Good/images
       - Copy Mask from CURATED  -> 1_Good/masks (Preserves original format)
    4. If NO (Bad):
       - Copy Image from TRAINING -> 0_Bad/images
       - Find Mask in BAD_MASKS_ROOT/{Structure}/w1 -> 0_Bad/masks (Looks for .tif only)
            
    Args:
        curated_root: Path to Curated data ({Struct}/image, {Struct}/label).
        training_root: Path to Training data ({Struct}/file.tif).
        bad_masks_root: Path to Bad Masks ({Struct}/w1/file.tiff).
        output_dir: Where to create the new dataset.
    """
    
    # Setup Output Directories
    good_img_dir = os.path.join(output_dir, "1_Good", "images")
    good_msk_dir = os.path.join(output_dir, "1_Good", "masks")
    bad_img_dir = os.path.join(output_dir, "0_Bad", "images")
    bad_msk_dir = os.path.join(output_dir, "0_Bad", "masks")

    for d in [good_img_dir, good_msk_dir, bad_img_dir, bad_msk_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Building Evaluator Dataset ---")
    print(f"Source Images:    {training_root}")
    print(f"Good Masks (Ref): {curated_root}")
    print(f"Bad Masks (Ref):  {bad_masks_root} (checking /w1 subfolders)")
    print(f"Output:           {output_dir}\n")

    # Get list of structure folders in Training
    training_structures = [d for d in os.listdir(training_root) if os.path.isdir(os.path.join(training_root, d))]
    
    good_count = 0
    bad_count = 0
    bad_mask_missing_count = 0
    good_mask_missing_count = 0

    for struct in training_structures:
        tr_struct_path = os.path.join(training_root, struct)
        
        # Paths for Good Masks (Curated)
        curated_img_path = os.path.join(curated_root, struct, "image")
        curated_lbl_path = os.path.join(curated_root, struct, "label")
        
        # Path for Bad Masks (Strictly looking in 'w1')
        # Structure: BadMasksRoot / Structure / w1
        bad_mask_w1_path = os.path.join(bad_masks_root, struct, "w1")

        # Check if this structure exists in Curated
        is_curated_struct = os.path.isdir(curated_img_path)
        
        # Load curated filenames for quick lookup
        curated_files_set = set()
        if is_curated_struct:
            curated_files_set = set(os.listdir(curated_img_path))

        # Iterate through Training files
        files = os.listdir(tr_struct_path)
        for f in files:
            src_img_full_path = os.path.join(tr_struct_path, f)
            
            if not os.path.isfile(src_img_full_path):
                continue

            # Base name for destination (Struct_Filename)
            dst_name_base = f"{struct}_{f}"
            file_stem, _ = os.path.splitext(f)

            # LOGIC CHECK: Is this file in Curated?
            if f in curated_files_set:
                # === CASE 1: GOOD (Class 1) ===
                # 1. Copy Image
                shutil.copy2(src_img_full_path, os.path.join(good_img_dir, dst_name_base))
                
                # 2. Find & Copy Mask (Curated/label)
                mask_found = False
                search_extensions = [os.path.splitext(f)[1], '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
                
                for ext in search_extensions:
                    test_name = file_stem + ext
                    src_mask_path = os.path.join(curated_lbl_path, test_name)
                    
                    if os.path.exists(src_mask_path):
                        dst_mask_name = f"{struct}_{file_stem}{ext}"
                        shutil.copy2(src_mask_path, os.path.join(good_msk_dir, dst_mask_name))
                        mask_found = True
                        good_count += 1
                        break
                
                if not mask_found:
                    print(f"[WARN] Good Image '{f}' missing mask in {curated_lbl_path}")
                    good_mask_missing_count += 1

            else:
                # === CASE 2: BAD (Class 0) ===
                # 1. Copy Image
                shutil.copy2(src_img_full_path, os.path.join(bad_img_dir, dst_name_base))
                
                # 2. Find & Copy Mask (BadMasksRoot/{Structure}/w1/file.tif)
                mask_found = False
                if os.path.isdir(bad_mask_w1_path):
                    # We strictly look for .tif or .tiff in the w1 folder
                    for ext in ['.tif', '.tiff']:
                        test_name = file_stem + ext
                        src_mask_path = os.path.join(bad_mask_w1_path, test_name)
                        
                        if os.path.exists(src_mask_path):
                            dst_mask_name = f"{struct}_{file_stem}{ext}"
                            shutil.copy2(src_mask_path, os.path.join(bad_msk_dir, dst_mask_name))
                            bad_count += 1
                            mask_found = True
                            break
                
                if not mask_found:
                    # print(f"[WARN] Bad Mask not found for '{f}' in '{bad_mask_w1_path}'")
                    bad_mask_missing_count += 1

    print(f"\nDataset Build Complete.")
    print(f"-> 1_Good: {good_count} pairs.")
    print(f"-> 0_Bad:  {bad_count} pairs.")
    
    if good_mask_missing_count > 0:
        print(f"[WARNING] {good_mask_missing_count} Good images were missing masks in Curated.")
    if bad_mask_missing_count > 0:
        print(f"[WARNING] {bad_mask_missing_count} Bad images were missing masks in '{bad_masks_root}/.../w1'.")


def count_structures_in_dataset(dataset_class_root):
    """
    Counts the number of images per Structure in a dataset folder.
    Assumes filenames are formatted as: "Structure_OriginalName.ext"
    
    Args:
        dataset_class_root: Path to the specific class folder 
                            (e.g., 'Evaluator_Dataset/1_Good' or 'Evaluator_Dataset/0_Bad')
    """
    
    # We count based on the 'images' folder, as it contains the source of truth
    images_dir = os.path.join(dataset_class_root, "images")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at '{images_dir}'")
        return

    print(f"--- Dataset Statistics ---")
    print(f"Scanning: {images_dir}\n")

    files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    
    structure_counts = Counter()
    
    for filename in files:
        # Logic: Filename is "Structure_OriginalName.tif"
        # We split by the first underscore only
        parts = filename.split('_', 1)
        
        if len(parts) > 1:
            structure_name = parts[0]
            structure_counts[structure_name] += 1
        else:
            print(f"[WARN] Filename '{filename}' does not match 'Structure_Name' pattern.")
            structure_counts["Unknown"] += 1

    # Print Results
    print(f"{'Structure':<20} | {'Count':<10}")
    print("-" * 35)
    
    total = 0
    for struct, count in sorted(structure_counts.items()):
        print(f"{struct:<20} | {count:<10}")
        total += count
        
    print("-" * 35)
    print(f"{'TOTAL':<20} | {total:<10}")

def clean_unpaired_images(bad_dataset_root, dry_run=True):
    """
    Scans the '0_Bad' dataset folder.
    Deletes images in 'images/' that do not have a matching mask in 'masks/'.
    
    Args:
        bad_dataset_root (str): Path to the '0_Bad' folder. 
                                Must contain 'images' and 'masks' subfolders.
        dry_run (bool): If True, only prints what would be deleted.
    """
    
    img_dir = os.path.join(bad_dataset_root, "images")
    msk_dir = os.path.join(bad_dataset_root, "masks")

    if not os.path.exists(img_dir) or not os.path.exists(msk_dir):
        print(f"Error: Could not find 'images' or 'masks' folders inside: {bad_dataset_root}")
        return

    print(f"--- Cleaning Unpaired Data ---")
    print(f"Target: {bad_dataset_root}")
    print(f"Mode:   {'DRY RUN (Safe)' if dry_run else 'LIVE EXECUTION (Destructive)'}\n")

    # 1. Index all available masks by their filename stem (ignoring extension)
    #    Example: 'UL_file1.png' -> 'UL_file1'
    mask_stems = set()
    try:
        mask_files = os.listdir(msk_dir)
        for f in mask_files:
            if os.path.isfile(os.path.join(msk_dir, f)):
                stem, _ = os.path.splitext(f)
                mask_stems.add(stem)
    except OSError as e:
        print(f"Error reading masks directory: {e}")
        return

    print(f"Found {len(mask_stems)} valid masks.")

    # 2. Check images against mask stems
    deleted_count = 0
    kept_count = 0
    
    try:
        image_files = os.listdir(img_dir)
        for f in image_files:
            img_path = os.path.join(img_dir, f)
            
            if not os.path.isfile(img_path):
                continue
                
            img_stem, _ = os.path.splitext(f)

            # CHECK: Does this image have a mask?
            if img_stem not in mask_stems:
                if dry_run:
                    print(f"[DRY RUN] Would delete: {f} (No matching mask found)")
                else:
                    try:
                        os.remove(img_path)
                        print(f"[DELETED] {f}")
                    except OSError as e:
                        print(f"[ERROR] Could not delete {f}: {e}")
                deleted_count += 1
            else:
                kept_count += 1

    except OSError as e:
        print(f"Error reading images directory: {e}")
        return

    print("\n" + "="*30)
    print(f"Cleanup Complete.")
    print(f"Images Kept:    {kept_count}")
    print(f"Images Deleted: {deleted_count} (or marked for deletion)")
    
    if dry_run and deleted_count > 0:
        print("\nTo actually delete these files, set 'dry_run=False' in the script.")

def create_scaled_dataset(source_root, output_root, percentage=0.1):
    """
    Creates a smaller version of the Evaluator Dataset by randomly sampling 
    a percentage of the data from both 1_Good and 0_Bad classes.
    
    Args:
        source_root (str): Path to the full dataset (contains 1_Good, 0_Bad).
        output_root (str): Path where the mini dataset will be created.
        percentage (float): Fraction of data to keep (e.g., 0.1 for 10%).
    """
    
    if not os.path.exists(source_root):
        print(f"Error: Source dataset '{source_root}' not found.")
        return

    # Structure definition
    classes = ["1_Good", "0_Bad"]
    subfolders = ["images", "masks"]

    print(f"--- Creating Scaled Dataset ({percentage*100}%) ---")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}\n")

    total_copied = 0

    for cls in classes:
        src_class_path = os.path.join(source_root, cls)
        dst_class_path = os.path.join(output_root, cls)
        
        src_img_dir = os.path.join(src_class_path, "images")
        src_msk_dir = os.path.join(src_class_path, "masks")
        
        dst_img_dir = os.path.join(dst_class_path, "images")
        dst_msk_dir = os.path.join(dst_class_path, "masks")

        # Create output directories
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_msk_dir, exist_ok=True)

        # Check if source class exists
        if not os.path.exists(src_img_dir):
            print(f"Skipping {cls}: 'images' folder not found.")
            continue

        # Get list of all images
        all_images = [f for f in os.listdir(src_img_dir) if os.path.isfile(os.path.join(src_img_dir, f))]
        
        # Calculate sample size
        total_files = len(all_images)
        sample_size = math.ceil(total_files * percentage)
        
        print(f"Processing {cls}: {total_files} files found -> Sampling {sample_size}...")

        # Randomly select files
        selected_images = random.sample(all_images, sample_size)
        
        # Build a map of mask files for quick lookup (Stem -> Full Filename)
        # This handles cases where image is .tif but mask is .png
        mask_map = {}
        if os.path.exists(src_msk_dir):
            for f in os.listdir(src_msk_dir):
                stem, _ = os.path.splitext(f)
                mask_map[stem] = f

        class_copied_count = 0
        
        for img_name in selected_images:
            img_stem, _ = os.path.splitext(img_name)
            
            # 1. Copy Image
            src_img = os.path.join(src_img_dir, img_name)
            dst_img = os.path.join(dst_img_dir, img_name)
            shutil.copy2(src_img, dst_img)
            
            # 2. Find and Copy Mask
            if img_stem in mask_map:
                mask_name = mask_map[img_stem]
                src_msk = os.path.join(src_msk_dir, mask_name)
                dst_msk = os.path.join(dst_msk_dir, mask_name)
                shutil.copy2(src_msk, dst_msk)
                class_copied_count += 1
            else:
                print(f"   [WARN] Mask missing for selected image: {img_name}")
                # We typically still copy the image, or you could choose to delete it here
        
        total_copied += class_copied_count
        print(f"   -> Copied {class_copied_count} pairs.")

    print("\n" + "="*30)
    print(f"Dataset Scaling Complete.")
    print(f"Total pairs created: {total_copied}")
    print(f"Location: {output_root}")

def build_balanced_dataset(source_root, output_root):
    """
    Creates a NEW dataset at 'output_root' that is balanced.
    
    Logic:
    1. Count '1_Good' and '0_Bad' in 'source_root'.
    2. Determine which class is smaller (Minority).
    3. Copy ALL files from the Minority class to 'output_root'.
    4. Randomly sample the Majority class to match the Minority count.
    5. Copy the sampled Majority files to 'output_root'.
    """
    
    # Define source paths
    src_good_dir = os.path.join(source_root, "1_Good")
    src_bad_dir  = os.path.join(source_root, "0_Bad")

    # Define output paths
    out_good_img = os.path.join(output_root, "1_Good", "images")
    out_good_msk = os.path.join(output_root, "1_Good", "masks")
    out_bad_img  = os.path.join(output_root, "0_Bad", "images")
    out_bad_msk  = os.path.join(output_root, "0_Bad", "masks")

    # Verify source exists
    if not os.path.exists(src_good_dir) or not os.path.exists(src_bad_dir):
        print(f"Error: Source dataset structure incorrect. Could not find 1_Good/0_Bad in {source_root}")
        return

    # Create output directories
    for d in [out_good_img, out_good_msk, out_bad_img, out_bad_msk]:
        os.makedirs(d, exist_ok=True)

    print(f"--- Building Balanced Dataset ---")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}\n")

    # 1. Index files in source
    def get_file_pairs(class_root):
        """Returns list of (img_filename, mask_filename) tuples"""
        img_dir = os.path.join(class_root, "images")
        msk_dir = os.path.join(class_root, "masks")
        
        pairs = []
        try:
            images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
            
            # Map mask stems for quick lookup
            masks = os.listdir(msk_dir)
            mask_map = {os.path.splitext(m)[0]: m for m in masks}
            
            for img in images:
                stem, _ = os.path.splitext(img)
                # Only include valid pairs (Image + Mask)
                if stem in mask_map:
                    pairs.append((img, mask_map[stem]))
        except OSError:
            pass
        return pairs

    print("Indexing source files...")
    good_pairs = get_file_pairs(src_good_dir)
    bad_pairs  = get_file_pairs(src_bad_dir)

    count_good = len(good_pairs)
    count_bad = len(bad_pairs)

    print(f"Found Pairs:")
    print(f"  1_Good: {count_good}")
    print(f"  0_Bad:  {count_bad}")

    if count_good == 0 or count_bad == 0:
        print("Error: One of the classes is empty. Cannot balance.")
        return

    # 2. Determine target count (size of minority class)
    target_count = min(count_good, count_bad)
    print(f"\nTarget Balance Count: {target_count} per class")

    # 3. Select files to copy
    # Minority class: Take all
    # Majority class: Random sample
    if count_good > count_bad:
        selected_bad  = bad_pairs # Take all bad
        selected_good = random.sample(good_pairs, target_count) # Sample good
        print(f"Sampling '1_Good' down to {target_count}...")
    else:
        selected_good = good_pairs # Take all good
        selected_bad  = random.sample(bad_pairs, target_count) # Sample bad
        print(f"Sampling '0_Bad' down to {target_count}...")

    # 4. Perform Copy
    def copy_pairs(pairs_list, src_root_class, dst_img_dir, dst_msk_dir):
        count = 0
        src_img_dir = os.path.join(src_root_class, "images")
        src_msk_dir = os.path.join(src_root_class, "masks")
        
        for img_name, msk_name in pairs_list:
            try:
                # Copy Image
                shutil.copy2(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
                # Copy Mask
                shutil.copy2(os.path.join(src_msk_dir, msk_name), os.path.join(dst_msk_dir, msk_name))
                count += 1
                
                if count % 100 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            except Exception as e:
                print(f"[Error] Copying {img_name}: {e}")
        return count

    print("\nCopying '1_Good' data...")
    copied_good = copy_pairs(selected_good, src_good_dir, out_good_img, out_good_msk)
    
    print("\nCopying '0_Bad' data...")
    copied_bad = copy_pairs(selected_bad, src_bad_dir, out_bad_img, out_bad_msk)

    print(f"\n\n[SUCCESS] New balanced dataset created at:")
    print(f"{output_root}")
    print(f"Final Counts: 1_Good={copied_good}, 0_Bad={copied_bad}")

if __name__ == "__main__":
    # ================= CONFIGURATION =================
    Bad = convert_path(r"C:\Repo\Metrology\data\UNET-6 mask")

    CURATED = convert_path(r"C:\Repo\Metrology\Raw_Data\Curated")
    TRAINING = convert_path(r"C:\Repo\Metrology\Raw_Data\Training")
    TEST = convert_path(r"C:\Repo\Metrology\Raw_Data\Test")
    
    d1 = convert_path(r"C:\Repo\Metrology\Evaluator_Dataset\1_Good")
    d2 = convert_path(r"C:\Repo\Metrology\Evaluator_Dataset\0_Bad")

    d3 = convert_path(r"C:\Repo\Metrology\Evaluator_Dataset_Test\0_Bad")

    #build_evaluator_dataset(CURATED, TEST, Bad, "Evaluator_Dataset_Test")
    #clean_unpaired_images(d3, dry_run=True)









    
