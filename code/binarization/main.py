import img_proc

if __name__ == "__main__":
    img_path = r"" # Image file Path
    save_path = "" # Save file Path
    mask = img_proc.binarize_img(img_path)
    img_proc.save_mask_png(mask, save_path)