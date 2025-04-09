import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

LABEL_MAPPING = {'AML': 0, 'ALL': 1, 'Normal': 2}


def load_and_preprocess(path):
    df = pd.read_csv(path, index_col=0, low_memory=False).T

    # Extract labels
    labels = df['Cancer'].map(LABEL_MAPPING).dropna()  # Dropping rows without a valid label

    # Extract gene expression data
    gene_expressions = df.drop(['Cancer', 'Preservation Method', 'Tissue Type', 'Tumor Descriptor', 'Specimen Type'], 
                               axis=1, errors='ignore')

    # Convert gene expression data to float32, ignore non-numeric columns
    try:
        gene_expressions = gene_expressions.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print("Error during conversion of gene expression data to float:", e)
    
    # Drop any columns that couldn't be converted to numeric
    gene_expressions = gene_expressions.dropna(axis=1)
    print(f"Gene expression data reduced to {gene_expressions.shape[1]} columns after removing non-numeric data.")
    
    # Convert to float32 for training
    gene_expressions = gene_expressions.astype(np.float32)

    return gene_expressions.values, labels.loc[gene_expressions.index].values



@config_enumerate
def model(gene_data, labels=None):
    # Hyperpriors
    alpha = pyro.sample("alpha", dist.Gamma(1, 1))

    # HDPMM components
    with pyro.plate("components", 200):  # Increased number of clusters for better separation
        stick_breaking = pyro.sample("beta", dist.Beta(1, alpha))
        cluster_means = pyro.sample("means", dist.Normal(0, 1).expand([500]).to_event(1))

    with pyro.plate("data", gene_data.shape[0]):
        # Cluster assignment
        weights = stick_breaking * torch.cumprod(1 - stick_breaking, dim=0)
        assignment = pyro.sample("assignment", dist.Categorical(weights))

        # Observation model
        pyro.sample("obs", dist.Normal(cluster_means[assignment], 0.1).to_event(1), 
                  obs=gene_data)


def guide(gene_data, labels=None):
    # Variational parameters
    alpha_q = pyro.param("alpha_q", torch.tensor(1.0), constraint=dist.constraints.positive)

    pyro.sample("alpha", dist.Delta(alpha_q))

    with pyro.plate("components", 200):
        pyro.sample("beta", dist.Beta(torch.ones(200), alpha_q))
        pyro.sample("means", dist.Normal(torch.zeros(500), 0.1).to_event(1))


def train(train_path, val_path, test_path, num_epochs=200, batch_size=64, evaluate_model=True):
    print('Starting training process...')
    # Load and preprocess
    X_train, y_train = load_and_preprocess(train_path)
    print(f'Loaded training data with {X_train.shape[0]} samples and {X_train.shape[1]} features')
    X_val, y_val = load_and_preprocess(val_path)
    print(f'Loaded validation data with {X_val.shape[0]} samples and {X_val.shape[1]} features')
    X_test, y_test = load_and_preprocess(test_path)
    print(f'Loaded test data with {X_test.shape[0]} samples and {X_test.shape[1]} features')

    # Dimensionality reduction
    pca = IncrementalPCA(n_components=500)
    X_train = pca.fit_transform(X_train)
    print('PCA transformation applied to training data')
    X_val = pca.transform(X_val)
    print('PCA transformation applied to validation data')
    X_test = pca.transform(X_test)
    print('PCA transformation applied to test data')

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Dataset and loader
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    pyro.clear_param_store()
    print('Initializing the model...')
    optimizer = Adam({"lr": 0.001})
    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            loss = svi.step(x_batch, y_batch)
            total_loss += loss / len(x_batch)

        print(f"Epoch {epoch} | Loss: {total_loss:.2f}")


def evaluate(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        dataset = TensorDataset(test_data, test_labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        total = 0
        correct = 0

        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            predicted = torch.argmax(y_pred, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    train_path = "/work3/s214806/working_chunk_train.csv"
    val_path = "/work3/s214806/working_chunk_val.csv"
    test_path = "/work3/s214806/working_chunk_test.csv"

    train(train_path, val_path, test_path)

