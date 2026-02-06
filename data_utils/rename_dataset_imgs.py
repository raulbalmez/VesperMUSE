import os
import shutil

def multiply_files_with_custom_prefixes(source_folder, output_folder, prefixes):
    if not os.path.isdir(source_folder):
        print(f"Error: '{source_folder}' is not a valid directory.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)

        if not os.path.isfile(source_path):
            continue

        for prefix in prefixes:
            new_filename = f"{prefix}{filename}"
            target_path = os.path.join(output_folder, new_filename)

            shutil.copy2(source_path, target_path)
            print(f"Copied: {filename} → {new_filename}")

if __name__ == "__main__":
    source_folder = "/home/ancuti/Downloads/CBSD68-dataset-master/CBSD68/original_png"
    output_folder = "/home/ancuti/Downloads/CBSD68-dataset-master/CBSD68/originalx5"
    prefixes = ["noise_clean_10", "noise_clean_15", "noise_clean_25", "noise_clean_35", "noise_clean_50"]

    multiply_files_with_custom_prefixes(source_folder, output_folder, prefixes)
