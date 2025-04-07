import os
import pandas as pd
import random


def load_chunks(output_dir):
    """
    Load all chunks and collect sample IDs from each chunk.
    """
    chunk_files = [f for f in os.listdir(output_dir) if f.startswith('chunk_') and f.endswith('.csv')]
    all_samples = []

    for chunk_file in chunk_files:
        chunk_path = os.path.join(output_dir, chunk_file)
        try:
            # Read the chunk file with improved settings
            chunk_df = pd.read_csv(chunk_path, dtype=str, low_memory=False)
            print(f"Columns found in {chunk_file}: {list(chunk_df.columns)}")  # Debugging print
            
            # Remove metadata rows if present
            non_metadata_columns = [col for col in chunk_df.columns if not any(
                keyword in col.lower() for keyword in ['gene_id', 'tissue type', 'tumor descriptor', 'specimen type', 'preservation method'])]
            
            sample_ids = non_metadata_columns
            all_samples.extend(sample_ids)
            print(f"Loaded chunk: {chunk_file} with {len(sample_ids)} samples")
        except Exception as e:
            print(f"Error loading chunk {chunk_file}: {e}")
    return all_samples, chunk_files



def select_samples(all_samples, num_samples):
    """
    Randomly select a subset of samples.
    """
    return random.sample(all_samples, num_samples)


def assemble_working_chunk(selected_samples, chunk_files, output_dir, working_chunk_path):
    """
    Assemble the working chunk from the selected samples and save to disk.
    """
    working_chunk = pd.DataFrame()

    for chunk_file in chunk_files:
        chunk_path = os.path.join(output_dir, chunk_file)
        try:
            chunk_df = pd.read_csv(chunk_path, index_col=0, dtype=str, low_memory=False)
            selected_columns = [col for col in chunk_df.columns if col in selected_samples]
            if selected_columns:
                working_chunk = pd.concat([working_chunk, chunk_df[selected_columns]], axis=1)
        except Exception as e:
            print(f"Error processing chunk {chunk_file}: {e}")

    # Save the working chunk to disk
    working_chunk.to_csv(working_chunk_path)
    print(f"Working chunk saved to: {working_chunk_path}")


if __name__ == "__main__":
    # Paths
    output_dir = "/work3/s214806/chunked_results/"
    working_chunk_path = "/work3/s214806/working_chunk.csv"

    # Parameters
    num_samples = 500  # Number of samples to randomly select

    # Load chunks and gather samples
    all_samples, chunk_files = load_chunks(output_dir)

    # Select random samples
    selected_samples = select_samples(all_samples, num_samples)

    # Assemble working chunk
    assemble_working_chunk(selected_samples, chunk_files, output_dir, working_chunk_path)
