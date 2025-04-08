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
COVARIATES = ['Preservation Method']


def load_and_preprocess(path):
    df = pd.read_csv(path, index_col=0, low_memory=False).T

    # Extract labels
    labels = df['Cancer'].map(LABEL_MAPPING).dropna()  # Dropping rows without a valid label

    # Extract and process the 'Preservation Method' covariate
    if 'Preservation Method' in df.columns:
        cov_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        covariates = cov_encoder.fit_transform(df.loc[labels.index, ['Preservation Method']])
    else:
        raise ValueError("Column 'Preservation Method' is missing from the dataset.")

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

    return gene_expressions.values, covariates, labels.loc[gene_expressions.index].values



@config_enumerate
def model(gene_data, covariates, labels=None):
    # Hyperpriors
    alpha = pyro.sample("alpha", dist.Gamma(1, 1))
    cov_strength = pyro.sample("cov_strength", dist.HalfNormal(1))

    # Covariate hierarchy
    with pyro.plate("covariates", covariates.shape[1]):
        cov_effects = pyro.sample("cov_effects", dist.Normal(0, cov_strength).expand([covariates.shape[1]]).to_event(1))

    # HDPMM components
    with pyro.plate("components", 200):  # Increased number of clusters for better separation
        stick_breaking = pyro.sample("beta", dist.Beta(1, alpha))
        cluster_means = pyro.sample("means", dist.Normal(0, 1).expand([500]).to_event(1))

    cov_contribution = torch.mm(covariates, cov_effects.unsqueeze(-1)).squeeze()

    with pyro.plate("data", gene_data.shape[0]):
        # Cluster assignment
        weights = stick_breaking * torch.cumprod(1 - stick_breaking, dim=0)
        assignment = pyro.sample("assignment", dist.Categorical(weights))

        # Observation model
        cov_contribution = cov_contribution.unsqueeze(-1).expand(-1, cluster_means.size(1))
        pyro.sample("obs", dist.Normal(cluster_means[assignment] + cov_contribution, 0.1).to_event(1), 
                  obs=gene_data)



def guide(gene_data, covariates, labels=None):
    # Variational parameters
    alpha_q = pyro.param("alpha_q", torch.tensor(1.0), constraint=dist.constraints.positive)
    cov_strength_q = pyro.param("cov_strength_q", torch.tensor(1.0), 
                              constraint=dist.constraints.positive)

    pyro.sample("alpha", dist.Delta(alpha_q))
    pyro.sample("cov_strength", dist.Delta(cov_strength_q))

    with pyro.plate("components", 200):
        pyro.sample("beta", dist.Beta(torch.ones(200), alpha_q))
        pyro.sample("means", dist.Normal(torch.zeros(500), 0.1).to_event(1))

    pyro.sample("cov_effects", dist.Normal(torch.zeros(covariates.shape[1]), cov_strength_q))


def train(train_path, val_path, test_path, num_epochs=200, batch_size=64):
    print('Starting training process...')
    # Load and preprocess
    X_train, cov_train, y_train = load_and_preprocess(train_path)
    print(f'Loaded training data with {X_train.shape[0]} samples and {X_train.shape[1]} features')
    X_val, cov_val, y_val = load_and_preprocess(val_path)
    print(f'Loaded validation data with {X_val.shape[0]} samples and {X_val.shape[1]} features')
    X_test, cov_test, y_test = load_and_preprocess(test_path)
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
    cov_train = torch.tensor(cov_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Dataset and loader
    dataset = TensorDataset(X_train, cov_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    pyro.clear_param_store()
    print('Initializing the model...')
    optimizer = Adam({"lr": 0.001})
    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

    # Training loop
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        total_loss = 0
        for x_batch, cov_batch, y_batch in loader:
            loss = svi.step(x_batch, cov_batch, y_batch)
            total_loss += loss / len(x_batch)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss:.2f}")


if __name__ == "__main__":
    train_path = "/work3/s214806/working_chunk_train.csv"
    val_path = "/work3/s214806/working_chunk_val.csv"
    test_path = "/work3/s214806/working_chunk_test.csv"

    train(train_path, val_path, test_path)
