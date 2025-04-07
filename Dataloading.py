import os
import pandas as pd

# Paths
root_dir = "/work3/s214806/MBML Cancer Data/Genetic Expression/GDC Cancer MBML/"
sample_sheet_path = "/work3/s214806/MBML Cancer Data/Genetic Expression/gdc_sample_sheet.2025-04-03.tsv"
output_dir = "/work3/s214806/chunked_results/"  # Directory to store intermediate chunks
os.makedirs(output_dir, exist_ok=True)

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
chunk_size = 500  # Number of files to process per chunk
chunk_number = 1

# Iterate over folders to collect data
def print_progress(files_loaded):
    if files_loaded % 100 == 0:
        print(f"Loaded {files_loaded} files so far")

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".tsv"):
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
                print_progress(files_loaded)

                # Process a chunk
                if files_loaded % chunk_size == 0:
                    expression_matrix = pd.DataFrame.from_dict(expression_data, orient='columns')
                    expression_matrix.index = gene_ids

                    # Append metadata rows
                    metadata_rows = metadata.reindex(expression_matrix.columns).T
                    final_matrix = pd.concat([metadata_rows, expression_matrix])

                    # Save chunk to disk
                    chunk_path = os.path.join(output_dir, f"chunk_{chunk_number}.csv")
                    final_matrix.to_csv(chunk_path)
                    print(f"Finished and saved chunk {chunk_number} with {chunk_size} files")

                    # Reset for next chunk
                    expression_data.clear()
                    chunk_number += 1

            except Exception as e:
                print(f"Error reading {file_id}: {e}")
                continue

# Save remaining files in the last chunk
if expression_data:
    expression_matrix = pd.DataFrame.from_dict(expression_data, orient='columns')
    expression_matrix.index = gene_ids
    metadata_rows = metadata.reindex(expression_matrix.columns).T
    final_matrix = pd.concat([metadata_rows, expression_matrix])
    chunk_path = os.path.join(output_dir, f"chunk_{chunk_number}.csv")
    final_matrix.to_csv(chunk_path)
    print(f"Finished and saved final chunk {chunk_number} with {len(expression_data)} files")

print("All files have been processed and saved in chunks.")