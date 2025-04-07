import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def load_chunks(output_dir):
    chunk_files = [f for f in os.listdir(output_dir) if f.startswith('chunk_') and f.endswith('.csv')]
    all_samples = []
    cancer_info = {}

    for chunk_file in chunk_files:
        chunk_path = os.path.join(output_dir, chunk_file)
        try:
            chunk_df = pd.read_csv(chunk_path, dtype=str, low_memory=False, index_col=0)
            if 'Cancer' in chunk_df.index:
                cancer_info.update(chunk_df.loc['Cancer'].to_dict())

            non_metadata_columns = [col for col in chunk_df.columns if not any(
                keyword in col.lower() for keyword in ['gene_id', 'tissue type', 'tumor descriptor', 'specimen type', 'preservation method', 'cancer'])]

            all_samples.extend(non_metadata_columns)
            print(f"Loaded chunk: {chunk_file} with {len(non_metadata_columns)} samples")
        except Exception as e:
            print(f"Error loading chunk {chunk_file}: {e}")

    return all_samples, chunk_files, cancer_info


def split_samples(all_samples, cancer_info):
    samples = pd.Series(all_samples).to_frame("SampleID")
    samples["Cancer"] = samples["SampleID"].map(cancer_info)

    train_samples, temp_samples = train_test_split(samples, test_size=0.3, random_state=42, stratify=samples["Cancer"])
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42, stratify=temp_samples["Cancer"])

    return train_samples["SampleID"].tolist(), val_samples["SampleID"].tolist(), test_samples["SampleID"].tolist()


def assemble_working_chunk(selected_samples, chunk_files, output_dir, working_chunk_path):
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

    working_chunk.to_csv(working_chunk_path)
    print(f"Working chunk saved to: {working_chunk_path}")


if __name__ == "__main__":
    output_dir = "/work3/s214806/chunked_results/"
    working_chunk_train = "/work3/s214806/working_chunk_train.csv"
    working_chunk_val = "/work3/s214806/working_chunk_val.csv"
    working_chunk_test = "/work3/s214806/working_chunk_test.csv"

    all_samples, chunk_files, cancer_info = load_chunks(output_dir)
    train_samples, val_samples, test_samples = split_samples(all_samples, cancer_info)

    assemble_working_chunk(train_samples, chunk_files, output_dir, working_chunk_train)
    assemble_working_chunk(val_samples, chunk_files, output_dir, working_chunk_val)
    assemble_working_chunk(test_samples, chunk_files, output_dir, working_chunk_test)
