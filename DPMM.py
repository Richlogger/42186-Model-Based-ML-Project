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



def preprocess_data(working_chunk_path):
    # Load the working chunk CSV file
    data = pd.read_csv(working_chunk_path, index_col=0)

    # Extract sample names
    sample_names = data.columns[6:]  # Assuming first 6 rows are metadata + labels

    # Extract covariates
    covariates = data.iloc[0:4, 6:].T  # Transpose for easier handling
    covariates.columns = ["Tissue Type", "Tumor Descriptor", "Specimen Type", "Preservation Method"]
    covariates.index = sample_names

    # Convert categorical covariates to numeric codes
    covariates_encoded = covariates.apply(lambda x: pd.factorize(x)[0]).values

    # Extract cancer labels
    labels = data.loc['Cancer', sample_names].map(LABEL_MAPPING).values  # Now using the 'Cancer' row label directly

    # Extract gene expression data
    gene_expression = data.iloc[6:, 6:].astype(float).T
    gene_expression.columns = data.iloc[6:, 0].values  # Set gene IDs as column names
    gene_expression.index = sample_names

    return torch.tensor(gene_expression.values, dtype=torch.float32), torch.tensor(covariates_encoded, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def model(expressions, covariates, labels=None):
    # Define hyperparameters
    num_genes = expressions.shape[1]
    num_clusters = 50
    num_covariates = covariates.shape[1]
    num_classes = 3

    # Top Layer: Stick-breaking process for clusters
    alpha = pyro.sample("alpha", dist.Gamma(2.0, 0.5))
    v = pyro.sample("v", dist.Beta(torch.ones(num_clusters), alpha).expand([num_clusters]))

    # Stick-breaking construction to generate cluster weights
    with torch.no_grad():
        stick_segments = torch.cat([v, torch.tensor([1.0])])
        stick_weights = stick_segments[:-1] * torch.cumprod(1 - stick_segments[:-1], dim=0)

    theta = pyro.sample("theta", dist.Delta(stick_weights))

    # Middle Layer: Dirichlet Process Mixture Model (DPMM)
    with pyro.plate("clusters", num_clusters):
        cluster_means = pyro.sample("cluster_means", dist.Normal(torch.zeros(num_genes), torch.ones(num_genes)))
        cluster_covs = pyro.sample("cluster_covs", dist.LogNormal(torch.zeros(num_genes), torch.ones(num_genes)))

    # Observation Layer
    with pyro.plate("data", len(expressions)):
        assignment = pyro.sample("assignment", dist.Categorical(theta))
        expression_likelihood = dist.Normal(cluster_means[assignment], cluster_covs[assignment]).to_event(1)
        pyro.sample("obs", expression_likelihood, obs=expressions)

    # Classification Layer
    logits = nn.Linear(num_covariates, num_classes)(covariates)
    class_probs = nn.Softmax(dim=-1)(logits)

    if labels is not None:
        pyro.sample("labels", dist.Categorical(class_probs), obs=labels)


def guide(expressions, covariates, labels=None):
    num_clusters = 50
    num_genes = expressions.shape[1]

    alpha_q = pyro.param("alpha_q", torch.tensor(2.0), constraint=dist.constraints.positive)
    v_q = pyro.param("v_q", torch.ones(num_clusters) * 0.5, constraint=dist.constraints.unit_interval)

    cluster_means_q = pyro.param("cluster_means_q", torch.randn(num_clusters, num_genes))
    cluster_covs_q = pyro.param("cluster_covs_q", torch.ones(num_clusters, num_genes), constraint=dist.constraints.positive)

    pyro.sample("alpha", dist.Gamma(alpha_q, 1.0))
    pyro.sample("v", dist.Beta(v_q, alpha_q).expand([num_clusters]))

    with pyro.plate("clusters", num_clusters):
        pyro.sample("cluster_means", dist.Normal(cluster_means_q, 0.1))
        pyro.sample("cluster_covs", dist.LogNormal(cluster_covs_q, 0.1))


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


if __name__ == "__main__":
    working_chunk_path = "/work3/s214806/working_chunk.csv"
    train(working_chunk_path, num_epochs=50, lr=0.001)
