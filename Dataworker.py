import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class GeneExpressionDataset(Dataset):
    def __init__(self, csv_path):
        # Load the working chunk
        self.data = pd.read_csv(csv_path, index_col=0)
        
        # Extract metadata rows
        self.metadata = self.data.iloc[:4, :].T  # The first 4 rows are metadata
        self.gene_expression = self.data.iloc[4:, :].astype(float).T  # Remaining rows are gene expression
        
        # Encode metadata into numerical values (simplified for now)
        self.labels = self.metadata['Tumor Descriptor'].map({'Primary': 0, 'Recurrent': 1, 'None': 2}).values
        self.covariates = pd.get_dummies(self.metadata[['Specimen Type', 'Preservation Method']]).values

    def __len__(self):
        return len(self.gene_expression)

    def __getitem__(self, idx):
        expression = torch.tensor(self.gene_expression.iloc[idx].values, dtype=torch.float32)
        covariates = torch.tensor(self.covariates[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return expression, covariates, label


def get_dataloader(csv_path, batch_size=32, shuffle=True):
    dataset = GeneExpressionDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    csv_path = "/work3/s214806/working_chunk.csv"
    dataloader = get_dataloader(csv_path)

    for batch_idx, (expressions, covariates, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Expressions {expressions.shape}, Covariates {covariates.shape}, Labels {labels.shape}")
        break  # Test with only one batch