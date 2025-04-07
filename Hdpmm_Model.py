import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


LABEL_MAPPING = {'AML': 0, 'ALL': 1, 'Normal': 2}


def load_data(train_path, val_path, test_path):
    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    # Extract metadata
    cancer_train = train_data.loc['Cancer'].map(LABEL_MAPPING)
    cancer_val = val_data.loc['Cancer'].map(LABEL_MAPPING)
    cancer_test = test_data.loc['Cancer'].map(LABEL_MAPPING)

    # Remove metadata rows
    expressions_train = train_data.drop(['Cancer', 'Tissue Type', 'Tumor Descriptor', 'Specimen Type', 'Preservation Method'], axis=0)
    expressions_val = val_data.drop(['Cancer', 'Tissue Type', 'Tumor Descriptor', 'Specimen Type', 'Preservation Method'], axis=0)
    expressions_test = test_data.drop(['Cancer', 'Tissue Type', 'Tumor Descriptor', 'Specimen Type', 'Preservation Method'], axis=0)

    return expressions_train, expressions_val, expressions_test, cancer_train, cancer_val, cancer_test


def train(train_path, val_path, test_path, num_epochs=int(os.getenv('NUM_EPOCHS', 100)), lr=0.001):
    expressions_train, expressions_val, expressions_test, cancer_train, cancer_val, cancer_test = load_data(train_path, val_path, test_path)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(expressions_train.values.T, dtype=torch.float32)
    X_val = torch.tensor(expressions_val.values.T, dtype=torch.float32)
    X_test = torch.tensor(expressions_test.values.T, dtype=torch.float32)
    y_train = torch.tensor(cancer_train.values, dtype=torch.long)
    y_val = torch.tensor(cancer_val.values, dtype=torch.long)
    y_test = torch.tensor(cancer_test.values, dtype=torch.long)

    # Baseline Model Training (Logistic Regression)
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(expressions_train.T, cancer_train)
    baseline_pred = baseline_model.predict(expressions_test.T)
    baseline_accuracy = accuracy_score(cancer_test, baseline_pred)
    print(f"Baseline Model Accuracy: {baseline_accuracy * 100:.2f}%")

    num_clusters = 50
    num_genes = X_train.shape[1]

    def stick_breaking(v):
        v_cumprod = torch.cumprod(1 - v, dim=0)
        weights = torch.cat([v[0:1], v[1:] * v_cumprod[:-1]])
        return weights

    def model(X):
        alpha = pyro.sample("alpha", dist.Gamma(1.0, 1.0))
        with pyro.plate("clusters", num_clusters):
            v = pyro.sample("v", dist.Beta(torch.ones(num_clusters), alpha))
            cluster_means = pyro.sample("cluster_means", dist.Normal(0.0, 1.0).expand([num_genes]))

        theta = stick_breaking(v)

        with pyro.plate("data", X.shape[0]):
            assignment = pyro.sample("assignment", dist.Categorical(theta))
            pyro.sample("obs", dist.Normal(cluster_means[assignment], 0.1).to_event(1), obs=X)

    def guide(X):
        alpha_q = pyro.param("alpha_q", torch.tensor(1.0), constraint=dist.constraints.positive)
        v_q = pyro.param("v_q", torch.ones(num_clusters), constraint=dist.constraints.unit_interval)
        cluster_means_q = pyro.param("cluster_means_q", torch.randn(num_clusters, num_genes))

        pyro.sample("alpha", dist.Gamma(alpha_q, 1.0))
        with pyro.plate("clusters", num_clusters):
            pyro.sample("v", dist.Beta(v_q, torch.ones(num_clusters)))
            pyro.sample("cluster_means", dist.Normal(cluster_means_q, 0.1))

    pyro.clear_param_store()

    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for epoch in range(num_epochs):
        loss = svi.step(X_train)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss}")

    print("Training complete.")

    val_loss = svi.evaluate_loss(X_val)
    print(f"Validation Loss: {val_loss}")


if __name__ == "__main__":
    train_path = "/work3/s214806/working_chunk_train.csv"
    val_path = "/work3/s214806/working_chunk_val.csv"
    test_path = "/work3/s214806/working_chunk_test.csv"

    train(train_path, val_path, test_path)
