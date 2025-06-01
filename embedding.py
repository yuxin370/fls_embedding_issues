import os
from glob import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Specify the directory to process
    data_dir = "./flower_photos/daisy"
    assert os.path.isdir(data_dir), f"Directory {data_dir} does not exist or is not a folder."

    # 2. Define image preprocessing (consistent with training/validation)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3. Traverse all image files in the directory (only process .jpg/.jpeg/.png)
    img_paths = (
        glob(os.path.join(data_dir, "*.jpg"))
        + glob(os.path.join(data_dir, "*.jpeg"))
        + glob(os.path.join(data_dir, "*.png"))
    )
    img_paths.sort()  # Maintain order for reproducibility

    if len(img_paths) == 0:
        print(f"No jpg/jpeg/png images found in {data_dir}.")
        return

    # 4. Prepare saving directory
    os.makedirs("embedding", exist_ok=True)

    # ----------------------------
    # A) Single column format: Flatten each image into a 150528×1 vector → Final shape (150528, N_images)
    # ----------------------------
    embeddings_cols = []  # List elements are 1D arrays with shape=(150528,)

    for img_path in img_paths:
        # 4.1 Read image, preprocess, add batch dimension
        image = Image.open(img_path).convert("RGB")
        img_tensor = data_transform(image)                # → (3,224,224)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)   # → (1,3,224,224)

        # 4.2 Move to CPU, convert to NumPy, flatten into (150528,)
        arr = img_tensor.cpu().numpy().reshape(-1)        # 1*3*224*224 = 150528
        embeddings_cols.append(arr)

    # 5. Stack all column vectors along the "column" direction → shape = (150528, N_images)
    all_embeddings_cols = np.stack(embeddings_cols, axis=1)  # Each column corresponds to an image

    # 6. Save as CSV (each column represents an image), delimiter '|', keep 6 decimal places, no scientific notation
    np.savetxt(
        "embedding/all_embeddings_columns.csv",
        all_embeddings_cols,
        fmt="%.6f",
        delimiter="|"
    )

    # 7. Save as Parquet: DataFrame will recognize each column as a single image
    df_cols = pd.DataFrame(all_embeddings_cols,
                           columns=[os.path.splitext(os.path.basename(p))[0] for p in img_paths])
    df_cols.to_parquet("embedding/all_embeddings_columns.parquet", index=False)

    print(f"[Single column format] Processing complete: {len(img_paths)} images, "
          f"vector length = {all_embeddings_cols.shape[0]}, "
          "saved to:\n"
          "  - embedding/all_embeddings_columns.csv\n"
          "  - embedding/all_embeddings_columns.parquet\n")

    # ----------------------------
    # B) Three column format: Each image split into (R,G,B) → reshape into (50176,3) → Final horizontally concatenated (50176, 3*N_images)
    # ----------------------------
    embeddings_rgb = []   # List elements are 2D arrays with shape=(50176,3)
    colnames_rgb = []     # Column names list for Parquet/CSV

    for img_path in img_paths:
        # 4.1 Read image, preprocess, add batch dimension
        image = Image.open(img_path).convert("RGB")
        img_tensor = data_transform(image)                # → (3,224,224)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)   # → (1,3,224,224)

        # 4.2 Move to CPU, convert to NumPy, move channels to the last dimension → (1,224,224,3) → reshape into (50176,3)
        arr_rgb = (
            img_tensor
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)      # (1,224,224,3)
            .reshape(-1, 3)             # 224*224 = 50176 rows, 3 columns
        )
        embeddings_rgb.append(arr_rgb)

        # Generate 3 column names for this image, e.g., "flower1_R","flower1_G","flower1_B"
        base = os.path.splitext(os.path.basename(img_path))[0]
        colnames_rgb += [f"{base}_R", f"{base}_G", f"{base}_B"]

    # 5. Concatenate all (50176,3) small matrices horizontally → (50176, 3*N_images)
    all_embeddings_rgb = np.hstack(embeddings_rgb)

    # 6. Save as CSV (each image occupies 3 columns, column order is [img1_R,img1_G,img1_B, img2_R,img2_G,img2_B,...])
    np.savetxt(
        "embedding/all_embeddings_rgb_columns.csv",
        all_embeddings_rgb,
        fmt="%.6f",
        delimiter="|"
    )

    # 7. Save as Parquet: DataFrame requires passing column names
    df_rgb = pd.DataFrame(all_embeddings_rgb, columns=colnames_rgb)
    df_rgb.to_parquet("embedding/all_embeddings_rgb_columns.parquet", index=False)

    print(f"[Three column format] Processing complete: {len(img_paths)} images, "
          f"rows per (R,G,B) image = {all_embeddings_rgb.shape[0]}, "
          f"total number of columns = {all_embeddings_rgb.shape[1]}, "
          "saved to:\n"
          "  - embedding/all_embeddings_rgb_columns.csv\n"
          "  - embedding/all_embeddings_rgb_columns.parquet\n")


if __name__ == "__main__":
    main()
