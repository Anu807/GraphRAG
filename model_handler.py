import os

import pandas as pd
import streamlit as st

from modules.feature_extraction import extract_features_for_pair


@st.cache_resource
def load_model(model_choice, selected_db, _onto_data=None, _review_year=None):
    """Load a trained model for the selected database and model type."""
    print("Starting load_model")

    try:
        import torch
        import torch.nn as nn
        torch_ok = True
    except ImportError:
        torch = None
        nn = None
        torch_ok = False

    model_files = {
        "Aminer": {
            "Logistic Regression (LR)": "logistic_regression_model11989A465-prob.pkl",
            "Decision Tree (DT)": "DecisionTreeClassifier_model1989A465-prob.pkl",
            "Random Forest (RF)": "RandomForestClassifier1-1989A465-prob.pkl",
            "Multi-Layer Perceptron (MLP)": "mlp_model_aminer.pth",
            "Graph Convolutional Network (GCN)": "gcn_model_aminer.pkl",
        },
        "Mendeley": {
            "Logistic Regression (LR)": "logistic_regression_model2019A5-prob.pkl",
            "Decision Tree (DT)": "DecisionTreeClassifier_model2019A5-prob.pkl",
            "Random Forest (RF)": "RandomForestClassifier-2019A5-prob.pkl",
            "Multi-Layer Perceptron (MLP)": "mlp_model_mendeley.pth",
            "Graph Convolutional Network (GCN)": "gcn_model_mendeley.pkl",
        },
    }

    model_name = model_files.get(selected_db, {}).get(model_choice)
    if not model_name:
        st.warning(f"The model '{model_choice}' is not supported for database '{selected_db}'.")
        return None

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", model_name))
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None

    if model_choice == "Multi-Layer Perceptron (MLP)":
        if not torch_ok:
            st.error("Could not load MLP because PyTorch is not installed.")
            return None

        class MLP(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                return self.network(x)

        class PyTorchMLPWrapper:
            def __init__(self, model_instance, input_size):
                self.model = model_instance
                self.model.eval()
                self.input_size = input_size

            def predict_proba(self, features_df):
                if features_df.shape[1] != self.input_size:
                    st.error(
                        f"Feature input size mismatch: {features_df.shape[1]} received; "
                        f"{self.input_size} required"
                    )
                    return torch.tensor([[0.0, 0.0]]).numpy()

                features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
                with torch.no_grad():
                    probabilities = torch.softmax(self.model(features_tensor), dim=1).numpy()
                return probabilities

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            if isinstance(checkpoint, nn.Module):
                model_instance = checkpoint
            else:
                model_instance = MLP(input_size=6)
                state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
                model_instance.load_state_dict(state_dict)
            return PyTorchMLPWrapper(model_instance, 6)
        except Exception as exc:
            st.error(f"Could not load the PyTorch model: {exc}. Please check the MLP structure and .pth file.")
            return None

    if model_choice == "Graph Convolutional Network (GCN)":
        if not torch_ok:
            st.error("Could not load GCN because PyTorch is not installed.")
            return None

        try:
            import torch_geometric
        except ImportError:
            st.error("PyTorch Geometric is not installed. The GCN model cannot be loaded.")
            return None

        class GCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                from torch_geometric.nn import GCNConv
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                return self.conv2(x, edge_index)

        class GCNWrapper:
            def __init__(self, model, ontology, review_year):
                self.model = model
                self.ontology = ontology
                self.review_year = review_year
                self.model.eval()

            def predict_proba(self, features_df):
                if not hasattr(features_df, "author_names_pair"):
                    st.error("GCNWrapper: The names of the two authors were not provided.")
                    return torch.tensor([[0.0, 0.0]]).numpy()

                author_a, author_b = features_df.author_names_pair
                node_a = extract_features_for_pair(
                    self.ontology, author_a, author_a, self.review_year, model_type="MLP"
                )
                node_b = extract_features_for_pair(
                    self.ontology, author_b, author_b, self.review_year, model_type="MLP"
                )
                if node_a is None or node_b is None:
                    st.error("GCNWrapper: Could not extract features for both authors.")
                    return torch.tensor([[0.0, 0.0]]).numpy()

                x = torch.tensor(pd.concat([node_a, node_b]).values, dtype=torch.float32)
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                with torch.no_grad():
                    logits = self.model(x, edge_index)
                    combined = logits[0] + logits[1]
                    probability = torch.softmax(combined, dim=0).numpy()
                return probability.reshape(1, -1)

        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            model = GCN(
                checkpoint.get("in_channels", 6),
                checkpoint.get("hidden_channels", 32),
                checkpoint.get("out_channels", 2),
            )
            model.load_state_dict(checkpoint["state_dict"])

            if _onto_data is None or _review_year is None:
                st.error("GCN prediction requires ontology data and a review year.")
                return None

            return GCNWrapper(model, _onto_data, _review_year)
        except Exception as exc:
            st.error(f"Could not load the GCN model: {exc}. Please check the GCN model file.")
            return None

    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Could not load the model with joblib: {exc}")
        return None


def predict_link_prob(features_df, model):
    """Use the loaded model to predict the collaboration probability."""
    try:
        if not hasattr(model, "predict_proba"):
            st.error("The selected model does not support probability prediction (predict_proba).")
            return 0.0
        return float(model.predict_proba(features_df)[:, 1][0])
    except Exception as exc:
        st.error(f"An error occurred during prediction: {exc}")
        return 0.0


def suggest_collaborators(search_author_name, model, author_names, onto_data, review_year, num_suggestions=20):
    """Use the selected model to suggest potential collaborators."""
    if not author_names or not model:
        return pd.DataFrame()

    suggestions = []
    model_type = "GCN" if hasattr(model, "in_channels") else ""

    for other_author in author_names:
        if other_author == search_author_name:
            continue
        features_df = extract_features_for_pair(
            onto_data,
            search_author_name,
            other_author,
            review_year,
            model_type=model_type,
        )
        if features_df is not None:
            if model_type == "GCN":
                setattr(features_df, "author_names_pair", (search_author_name, other_author))
            probability = predict_link_prob(features_df, model)
            suggestions.append({
                "Author Name": other_author,
                "Collaboration Probability": probability,
            })

    suggestions_df = pd.DataFrame(suggestions)
    if not suggestions_df.empty:
        suggestions_df = suggestions_df.sort_values(
            by="Collaboration Probability",
            ascending=False,
        ).head(num_suggestions).reset_index(drop=True)
        suggestions_df["Collaboration Probability"] = suggestions_df["Collaboration Probability"].apply(
            lambda value: f"{value * 100:.2f}%"
        )
    return suggestions_df
