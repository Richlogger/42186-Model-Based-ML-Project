import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import numpy as np
import os


# Define label mapping
LABEL_MAPPING = {'AML': 0, 'ALL': 1, 'Normal': 2}


# Preprocessing the input data
# This function loads the working_chunk.csv file, extracts covariates, labels, and gene expressions.
# It also encodes categorical covariates into numerical values for use in the model.
def preprocess_data(working_chunk_path):
    # Load the working chunk CSV file
    data = pd.read_csv(working_chunk_path, index_col=0, low_memory=False)

    # Extract sample names
    sample_names = data.columns[6:]  # Assuming first 6 rows are metadata + labels

    # Extract covariates
    covariates = data.iloc[0:4, 6:].T  # Transpose for easier handling
    covariates.columns = ['Tissue Type', 'Tumor Descriptor', 'Specimen Type', 'Preservation Method']
    covariates.index = sample_names

    # Convert categorical covariates to numeric codes
    covariates_encoded = covariates.apply(lambda x: pd.factorize(x)[0]).values

    # Extract cancer labels
    labels = data.loc['Cancer', sample_names].map(LABEL_MAPPING).values

    # Extract gene expression data
    gene_expression = data.iloc[6:, 6:].astype(float).T
    gene_expression.columns = data.iloc[6:, 0].values
    gene_expression.index = sample_names

    return torch.tensor(gene_expression.values, dtype=torch.float32), torch.tensor(covariates_encoded, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# Stick-breaking function to generate cluster weights
# Uses a cumulative product to calculate weights from sampled beta distributions
def stick_breaking(v):
    v_cumprod = torch.cumprod(1 - v, dim=0)
    weights = torch.cat([v[0:1], v[1:] * v_cumprod[:-1]])
    return weights


# Define the hierarchical Dirichlet process model
# The model defines the clusters and their distributions using a stick-breaking process.
# Each sample is assigned to a cluster, which generates the gene expression data.
def model(expressions, covariates, labels=None):
    num_genes = expressions.shape[1]
    num_samples = expressions.shape[0]
    num_clusters = 50
    num_classes = 3

    alpha = pyro.sample("alpha", dist.Gamma(2.0, 0.5))

    with pyro.plate("clusters", num_clusters):
        v = pyro.sample("v", dist.Beta(torch.ones(num_clusters), alpha))
        cluster_means = pyro.sample("cluster_means", dist.Normal(torch.zeros(num_genes), torch.ones(num_genes)).to_event(1))
        cluster_covs = pyro.sample("cluster_covs", dist.LogNormal(torch.zeros(num_genes), torch.ones(num_genes)).to_event(1))

    # Compute cluster weights using stick-breaking process
    cluster_weights = stick_breaking(v)

    with pyro.plate("data", num_samples):
        assignment = pyro.sample("assignment", dist.Categorical(cluster_weights))
        expression_likelihood = dist.Normal(cluster_means[assignment], cluster_covs[assignment]).to_event(1)
        pyro.sample("obs", expression_likelihood, obs=expressions)


# Define the guide for variational inference
# This guide mirrors the model, defining variational distributions for all latent variables.
# It includes variational parameters for the stick-breaking process and cluster distributions.
def guide(expressions, covariates, labels=None):
    num_clusters = 50
    num_genes = expressions.shape[1]
    num_samples = expressions.shape[0]

    alpha_q = pyro.param("alpha_q", torch.tensor(2.0), constraint=dist.constraints.positive)
    v_q = pyro.param("v_q", torch.ones(num_clusters) * 0.5, constraint=dist.constraints.unit_interval)

    cluster_means_q = pyro.param("cluster_means_q", torch.randn(num_clusters, num_genes))
    cluster_covs_q = pyro.param("cluster_covs_q", torch.ones(num_clusters, num_genes), constraint=dist.constraints.positive)

    pyro.sample("alpha", dist.Gamma(alpha_q, 1.0))
    with pyro.plate("clusters", num_clusters):
        pyro.sample("v", dist.Beta(v_q, alpha_q))
        pyro.sample("cluster_means", dist.Normal(cluster_means_q, 0.1).to_event(1))
        pyro.sample("cluster_covs", dist.LogNormal(cluster_covs_q, 0.1).to_event(1))

    # Guide for assignment
    assignment_logits = pyro.param("assignment_logits", torch.randn(num_samples, num_clusters))
    with pyro.plate("data", num_samples):
        pyro.sample("assignment", dist.Categorical(logits=assignment_logits))


# Training function
# Runs stochastic variational inference (SVI) for a specified number of epochs.
# Prints the loss at regular intervals for monitoring progress.
def train(working_chunk_path, num_epochs=100, lr=0.001):
    expressions, covariates, labels = preprocess_data(working_chunk_path)

    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for epoch in range(num_epochs):
        loss = svi.step(expressions, covariates, labels)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss}")

    print("Training complete.")


# Evaluation function
# Simple linear classifier to evaluate how well the covariates predict the labels.
# Reports the accuracy as a performance measure.
def evaluate(expressions, covariates, labels):
    classifier = nn.Linear(covariates.shape[1], 3)
    logits = classifier(covariates)
    predictions = torch.argmax(logits, dim=-1)

    accuracy = (predictions == labels).float().mean().item()
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    working_chunk_path = "/work3/s214806/working_chunk.csv"
    train(working_chunk_path, num_epochs=50, lr=0.001)

