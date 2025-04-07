import os
import pandas as pd

# Paths
root_dir = "/work3/s214806/MBML Cancer Data/Genetic Expression/GDC Cancer MBML/"
sample_sheet_path = "/work3/s214806/MBML Cancer Data/Genetic Expression/gdc_sample_sheet.2025-04-03.tsv"
#output_path = "/work3/s214806/expression_matrix_test.csv"
output_path = "/work3/s214806/expression_matrix.csv"

# Load sample metadata
def load_sample_metadata(sample_sheet_path):
    metadata = pd.read_csv(sample_sheet_path, sep='\t')
    metadata = metadata[['File ID', 'Tissue Type', 'Tumor Descriptor', 'Specimen Type', 'Preservation Method']]
    metadata.set_index('File ID', inplace=True)
    return metadata

metadata = load_sample_metadata(sample_sheet_path)

# Prepare containers
expression_data = {}
gene_ids = None
files_loaded = 0  # Counter for progress tracking
max_files = 200  # Limiting the number of files to load

# Iterate over folders to collect data
def print_progress(files_loaded):
    if files_loaded % 100 == 0:
        print(f"Loaded {files_loaded} files so far")

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".tsv"):
            if files_loaded >= max_files:
                break
            file_id = os.path.basename(root)
            try:
                # Read the TSV file
                tsv_path = os.path.join(root, file)
                df = pd.read_csv(tsv_path, sep='\t', skiprows=[0, 2, 3, 4, 5])
                df = df[['gene_id', 'tpm_unstranded']]

                # Set gene order on first run
                if gene_ids is None:
                    gene_ids = df['gene_id']

                # Store expression data for this file_id
                expression_data[file_id] = df['tpm_unstranded'].values

                files_loaded += 1
                print_progress(files_loaded)  # Print progress every 100 files loaded

            except Exception as e:
                print(f"Error reading {file_id}: {e}")
                continue
    #if files_loaded >= max_files:
    #   break

# Create DataFrame
expression_matrix = pd.DataFrame.from_dict(expression_data, orient='columns')
expression_matrix.index = gene_ids

# Append metadata rows to the DataFrame
metadata_rows = metadata.reindex(expression_matrix.columns).T
final_matrix = pd.concat([metadata_rows, expression_matrix])

# Save to CSV
final_matrix.to_csv(output_path)
print(f"Expression matrix saved to: {output_path}")