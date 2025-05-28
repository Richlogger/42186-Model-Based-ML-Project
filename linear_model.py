import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from Hdpmm_Model import load_and_preprocess


def standardize_data(train_data, test_data):
    """
    Standardize the training and test datasets using Z-score normalization.
    """
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    return train_data_scaled, test_data_scaled


def train_logistic_regression(train_data, train_labels):
    """
    Train a multinomial logistic regression classifier.
    """
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100, random_state=42)
    clf.fit(train_data, train_labels)
    return clf


def evaluate_model(clf, test_data, test_labels, target_names):
    """
    Evaluate the trained model on the test dataset and print results.
    """
    test_preds = clf.predict(test_data)
    conf_matrix = confusion_matrix(test_labels, test_preds)
    class_report = classification_report(test_labels, test_preds, target_names=target_names)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)


def plot_feature_importance(clf, feature_names=None, top_n=20):
    """
    Plot the top N most important features based on logistic regression coefficients.
    """
    # Get the absolute values of the coefficients
    coef = np.abs(clf.coef_).mean(axis=0)  # Average across classes for multiclass
    if feature_names is None:
        feature_names = [f"Gene {i}" for i in range(len(coef))]

    # Get the top N features
    top_indices = np.argsort(coef)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_coef = coef[top_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_coef, color="skyblue")
    plt.xlabel("Feature Importance (Absolute Coefficient)")
    plt.title(f"Top {top_n} Most Important Features")
    plt.savefig("feature_importance.png")


class ANNClassifier(nn.Module):
    """
    Artificial Neural Network for classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_ann(train_data, train_labels, input_dim, hidden_dim, output_dim, epochs=50, lr=0.001):
    """
    Train an Artificial Neural Network classifier.
    """
    # Convert data to PyTorch tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    # Initialize the model, loss function, and optimizer
    model = ANNClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data_tensor)
        loss = criterion(outputs, train_labels_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    return model

def evaluate_ann(model, test_data, test_labels, target_names):
    """
    Evaluate the trained ANN model on the test dataset and print results.
    """
    # Convert test data to PyTorch tensors
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(test_data_tensor)
        _, test_preds = torch.max(outputs, 1)

    # Compute confusion matrix and classification report
    conf_matrix = confusion_matrix(test_labels_tensor.numpy(), test_preds.numpy())
    class_report = classification_report(test_labels_tensor.numpy(), test_preds.numpy(), target_names=target_names)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)



def main():
    # File paths
    train_path = "/work3/s214806/working_chunk_train.csv"
    test_path = "/work3/s214806/working_chunk_test.csv"

    # Load datasets using load_and_preprocess
    train_data, train_labels = load_and_preprocess(train_path)
    test_data, test_labels = load_and_preprocess(test_path)

    # Standardize datasets
    train_data_scaled, test_data_scaled = standardize_data(train_data, test_data)

    # Train logistic regression model
    clf = train_logistic_regression(train_data_scaled, train_labels)

    # Evaluate the model
    evaluate_model(clf, test_data_scaled, test_labels, target_names=["AML", "ALL", "Normal"])

    # Plot feature importance
    plot_feature_importance(clf, top_n=20)

   # Define model parameters
    input_dim = train_data_scaled.shape[1]
    hidden_dim = 128
    output_dim = len(np.unique(train_labels))  # Number of classes

    # Train ANN model
    model = train_ann(train_data_scaled, train_labels, input_dim, hidden_dim, output_dim, epochs=50, lr=0.001)

    # Evaluate the model
    evaluate_ann(model, test_data_scaled, test_labels, target_names=["AML", "ALL", "Normal"])



if __name__ == "__main__":
    main()