from PIL import Image
from pathlib import Path

folder = Path("L:/Lab_LucaC/Carina/figures/cycles")

# rename files for proper sorting

for f in folder.iterdir():
    for c in range(6):
        if f.name.startswith(f"cycle_{c}_") and f.suffix == ".png":
            # extract the number after cycle_0_
            num_str = f.stem.split("_")[-1]

            # zero-pad to 2 digits
            num_padded = num_str.zfill(2)

            new_name = f"cycle_{c}_{num_padded}.png"
            f.rename(folder / new_name)
            print(f"Renamed {f.name} -> {new_name}")


def images_to_pdf(image_dir, output_pdf):
    image_dir = Path(image_dir)

    # Get all image files (png, jpg, jpeg, tif...)
    img_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
    )

    if not img_files:
        raise ValueError("No image files found in the directory.")

    # Open all images
    images = [Image.open(f).convert("RGB") for f in img_files]

    # Save into a single PDF
    first_img = images[0]
    other_imgs = images[1:]

    first_img.save(
        output_pdf,
        save_all=True,
        append_images=other_imgs
    )

    print(f"PDF saved to: {output_pdf}")


images_to_pdf(folder, f"{folder}/cycles_plots.pdf")
