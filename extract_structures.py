import tarfile
import gzip
import os
import shutil
##############################################################################################
#### get a tar files and extract the first 500 pdb files  (dataset file is from alpha fold)
##############################################################################################
def extract_first_500_pdb(tar_path, extract_to="extracted_pdb_files", limit=500):
    os.makedirs(extract_to, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        temp_extract_dir = "temp_extracted_files"
        os.makedirs(temp_extract_dir, exist_ok=True)
        tar.extractall(path=temp_extract_dir)

    count = 0
    for root, dirs, files in os.walk(temp_extract_dir):
        for file in files:
            if count >= limit:
                print(f"Reached the limit of {limit} .pdb files.")
                break
            if file.endswith(".pdb.gz"):
                gz_file_path = os.path.join(root, file)
                with gzip.open(gz_file_path, "rb") as gz_file:
                    pdb_filename = file[:-3]  
                    pdb_output_path = os.path.join(extract_to, pdb_filename)
                    with open(pdb_output_path, "wb") as pdb_file:
                        shutil.copyfileobj(gz_file, pdb_file)
                        print(f"Extracted: {pdb_filename}")
                        count += 1
                        if count >= limit:
                            break

    shutil.rmtree(temp_extract_dir)
    print(f"Extracted {count} .pdb files to: {extract_to}")


########################################################
### Main function
########################################################
def main():
    tar_file_path = "UP000000799_192222_CAMJE_v4.tar"  
    extract_first_500_pdb(tar_file_path)

if __name__ == "__main__":
    main()