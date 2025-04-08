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
COVARIATES = ['Tissue Type', 'Tumor Descriptor', 'Specimen Type', 'Preservation Method']

def load_and_preprocess(path):
    df = pd.read_csv(path, index_col=0).T
    labels = df['Cancer'].map(LABEL_MAPPING)
    
    # Process covariates
    cov_encoder = OneHotEncoder(handle_unknown='ignore')
    covariates = cov_encoder.fit_transform(df[COVARIATES]).toarray()
    
    # Process gene expressions
    genes = df.drop(['Cancer'] + COVARIATES, axis=1)
    genes = genes.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return genes.values, covariates, labels.values

class GeneEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)

@config_enumerate
def model(gene_data, covariates, labels=None):
    encoder = pyro.module("encoder")
    encoded_genes = encoder(gene_data)
    
    # Hyperpriors
    alpha = pyro.sample("alpha", dist.Gamma(1, 1))
    cov_strength = pyro.sample("cov_strength", dist.HalfNormal(1))
    
    # Covariate hierarchy
    with pyro.plate("covariates", covariates.shape[1]):
        cov_effects = pyro.sample("cov_effects", dist.Normal(0, cov_strength))
    
    # HDPMM components
    with pyro.plate("components", 100):
        stick_breaking = pyro.sample("beta", dist.Beta(1, alpha))
        cluster_means = pyro.sample("means", dist.Normal(0, 1).expand([128]).to_event(1))
    
    # Data processing
    encoded_genes = pyro.module("encoder").net(gene_data)
    cov_contribution = torch.mm(covariates, cov_effects.unsqueeze(-1)).squeeze()
    
    with pyro.plate("data", gene_data.shape[0]):
        # Cluster assignment
        weights = stick_breaking * torch.cumprod(1 - stick_breaking, dim=0)
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        
        # Observation model
        latent_rep = encoded_genes + cov_contribution
        pyro.sample("obs", dist.Normal(cluster_means[assignment], 0.1).to_event(1), 
                  obs=latent_rep)
        
        # Classification
        if labels is not None:
            with pyro.poutine.scale(scale=0.1):
                class_probs = pyro.sample("class_probs", 
                                        dist.Dirichlet(torch.ones(3)))
                pyro.sample("label", dist.Categorical(class_probs[assignment]), 
                          obs=labels)

def guide(gene_data, covariates, labels=None):
    # Amortized cluster assignment
    encoder = pyro.module("encoder")
    encoded = encoder(gene_data)
    cov_effect = pyro.param("cov_bias", torch.zeros(covariates.shape[1]))
    logits = torch.mm(covariates, cov_effect.unsqueeze(-1)).squeeze() + encoded.mean(-1)
    
    # Variational parameters
    alpha_q = pyro.param("alpha_q", torch.tensor(1.0), constraint=dist.constraints.positive)
    cov_strength_q = pyro.param("cov_strength_q", torch.tensor(1.0), 
                              constraint=dist.constraints.positive)
    
    pyro.sample("alpha", dist.Delta(alpha_q))
    pyro.sample("cov_strength", dist.Delta(cov_strength_q))
    
    with pyro.plate("components", 100):
        pyro.sample("beta", dist.Beta(torch.ones(100), alpha_q))
        pyro.sample("means", dist.Normal(torch.zeros(128), 0.1).to_event(1))
    
    pyro.sample("cov_effects", dist.Normal(torch.zeros(covariates.shape[1]), cov_strength_q))

def train(train_path, val_path, test_path, num_epochs=200, batch_size=64):
    # Load and preprocess
    X_train, cov_train, y_train = load_and_preprocess(train_path)
    X_val, cov_val, y_val = load_and_preprocess(val_path)
    X_test, cov_test, y_test = load_and_preprocess(test_path)
    
    # Dimensionality reduction
    pca = IncrementalPCA(n_components=500)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    
    encoder = GeneEncoder(input_dim=500, latent_dim=128)
    pyro.module("encoder", encoder)  # Register once

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    cov_train = torch.tensor(cov_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    # Dataset and loader
    dataset = TensorDataset(X_train, cov_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    pyro.clear_param_store()
    optimizer = Adam({"lr": 0.001})
    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, cov_batch, y_batch in loader:
            loss = svi.step(x_batch, cov_batch, y_batch)
            total_loss += loss / len(x_batch)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss:.2f}")
            
    # Evaluation
    def evaluate(X, cov, y):
        predictive = pyro.infer.Predictive(model, guide=guide, num_samples=100)
        samples = predictive(X, cov, None)
        pred_labels = samples["label"].mode(0).values
        return accuracy_score(y.numpy(), pred_labels.numpy())
    
    print(f"\nValidation Accuracy: {evaluate(X_val, cov_val, y_val):.2%}")
    print(f"Test Accuracy: {evaluate(X_test, cov_test, y_test):.2%}")

if __name__ == "__main__":
    train_path = "/work3/s214806/working_chunk_train.csv"
    val_path = "/work3/s214806/working_chunk_val.csv"
    test_path = "/work3/s214806/working_chunk_test.csv"
    
    train(train_path, val_path, test_path)
