import os
import zipfile

def extract_zip(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

if __name__ == "__main__":
    zip_file_path = 'src/SCSPAR24_Testdata.zip'
    extract_dir = 'src'
    extract_zip(zip_file_path, extract_dir)
