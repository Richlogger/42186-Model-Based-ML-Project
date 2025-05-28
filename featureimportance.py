import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam, Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import time




def compute_feature_importance_from_AW(W, A, alpha_mean=None, top_n=20):
    """
    Compute feature importance based on the interaction between A and W.
    If alpha_mean is provided, it scales W using ARD weights.

    Parameters:
    - W: numpy array, loading matrix (latent_dim x n_genes).
    - A: numpy array, classification weights (n_classes x latent_dim).
    - alpha_mean: numpy array or None, ARD precision values (latent_dim). Default is None.
    - top_n: int, number of top features to return.

    Returns:
    - top_indices: indices of the top N most important genes.
    - top_importance: importance values for the top N genes.
    """
    if alpha_mean is not None:
        # Scale W by the ARD precision (alpha_mean)
        W_scaled = W / np.sqrt(alpha_mean[:, None])  # Adjust W using ARD weights
    else:
        # Use W as-is if alpha_mean is not provided
        W_scaled = W

    # Compute the interaction between A and scaled W
    AW_scaled = A @ W_scaled  # Shape: [n_classes, n_genes]

    # Compute gene importance as the L2 norm across classes
    gene_importance = np.linalg.norm(AW_scaled, axis=0)  # Shape: [n_genes]

    # Sort genes by importance and select the top N
    top_indices = np.argsort(gene_importance)[-top_n:][::-1]  # Indices of top N genes in descending order
    top_importance = gene_importance[top_indices]

    return top_indices, top_importance


def plot_top_genes_from_AW(top_indices, top_importance, top_n=20, gene_names=None):
    """
    Plot the top N most important genes based on the interaction between W and A.

    Parameters:
    - top_indices: list of indices of the top N most important genes.
    - top_importance: list of importance values for the top N genes.
    - top_n: int, number of top genes to plot.
    - gene_names: list of gene names (optional). If None, generic names will be used.
    """
    # Use generic names if gene names are not provided
    if gene_names is None:
        top_features = [f"Gene {i}" for i in top_indices]
    else:
        top_features = [gene_names[i] for i in top_indices]

    # Plot the top N most important genes
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importance, color="skyblue")
    plt.xlabel("Feature Importance (L2 Norm of AW)")
    plt.title(f"Top {top_n} Most Important Genes")
    plt.show()

def plot_all_genes_from_AW(gene_importance, gene_names=None):
    """
    Plot the importance of all genes to visualize the distribution of weights.

    Parameters:
    - gene_importance: numpy array, importance values for all genes.
    - gene_names: list of gene names (optional). If None, generic names will be used.
    """
    n_genes = len(gene_importance)

    # Use generic names if gene names are not provided
    if gene_names is None:
        gene_names = [f"Gene {i}" for i in range(n_genes)]

    # Sort genes by importance
    sorted_indices = np.argsort(gene_importance)[::-1]  # Descending order
    sorted_gene_importance = gene_importance[sorted_indices]
    sorted_gene_names = [gene_names[i] for i in sorted_indices]

    # Plot all genes
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_genes), sorted_gene_importance, color="skyblue", alpha=0.7)
    plt.xlabel("Genes (Sorted by Importance)")
    plt.ylabel("Feature Importance (L2 Norm of AW)")
    plt.title("Feature Importance for All Genes")
    plt.show()


'''
if __name__ == "__main__":
    W = pyro.param("w_loc").detach().cpu().numpy()  # Shape: [latent_dim, n_genes]
    A = pyro.param("A").detach().cpu().numpy()      # Shape: [n_classes, latent_dim]
    top_n = 20
    gene_importance = np.linalg.norm(A @ W, axis=0)  # Compute importance for all genes


    # Compute feature importance
    top_indices, top_importance = compute_feature_importance_from_AW(W, A, top_n=top_n)

    # Plot the top genes
    plot_top_genes_from_AW(top_indices, top_importance, top_n=top_n)
    plot_all_genes_from_AW(gene_importance)

'''