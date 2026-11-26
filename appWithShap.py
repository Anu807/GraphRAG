import os
import random
import re

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from modules.data_handler import load_ontology, load_authors_list
from modules.feature_extraction import extract_features_for_pair
from modules.model_handler import load_model, predict_link_prob, suggest_collaborators


st.set_page_config(layout="wide", page_title="Academic GraphRAG with SHAP")

st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 30% !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ontology...")
def load_cached_ontology(path):
    if not path:
        return None
    try:
        return load_ontology(path)
    except Exception as exc:
        st.error(f"Could not load the ontology: {exc}. Please check the file.")
        return None


@st.cache_data(show_spinner="Loading the author list...")
def get_cached_author_list(ontology):
    if not ontology:
        return []
    try:
        return load_authors_list(ontology)
    except Exception as exc:
        st.error(f"Could not extract the author list: {exc}")
        return []


st.sidebar.title("Academic Data Management ⚙️")
st.sidebar.header("Select a Database")
try:
    databases = sorted(d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d)))
    selected_db = st.sidebar.selectbox("Please select a database:", databases)
except FileNotFoundError:
    st.sidebar.error("The 'data/' directory was not found. Please create it.")
    st.stop()

ontology_path = None
review_year = None
if selected_db:
    st.sidebar.header("Select an Ontology")
    ontologies_dir = os.path.join("data", selected_db, "ontologies")
    try:
        owl_files = sorted(f for f in os.listdir(ontologies_dir) if f.lower().endswith(".owl"))
        if not owl_files:
            st.sidebar.warning(f"No ontology files were found in '{ontologies_dir}'.")
        selected_ontology = st.sidebar.selectbox("Please select an ontology file:", owl_files)
        ontology_path = os.path.join(ontologies_dir, selected_ontology)
        year_match = re.search(r"\b\d{4}\b", selected_ontology)
        if year_match:
            review_year = int(year_match.group(0))
            st.sidebar.info(f"Review year: **{review_year}**")
        else:
            st.sidebar.warning("No year could be found in the ontology filename.")
    except FileNotFoundError:
        st.sidebar.error(f"The directory '{ontologies_dir}' does not exist.")

onto_data = load_cached_ontology(ontology_path)
authors_full_list = get_cached_author_list(onto_data) if onto_data else []
author_names = [author.name for author in authors_full_list]

st.sidebar.header("Select a Model")
model_choice = st.sidebar.radio("Choose a model", [
    "Logistic Regression (LR)",
    "Decision Tree (DT)",
    "Random Forest (RF)",
    "Multi-Layer Perceptron (MLP)",
    "Graph Convolutional Network (GCN)",
])
st.sidebar.header("Prediction Parameters")
probability_threshold = st.sidebar.slider("Probability threshold", 0.0, 1.0, 0.5)

link_tab, search_tab, admin_tab, report_tab = st.tabs([
    "Link Prediction", "Search and Suggestions", "Administration", "Export Report"
])

with link_tab:
    st.header("1. Link Prediction 🤝")
    input_method = st.radio("Choose an input method", ("Manual", "Random", "Batch from File"))

    author_a = None
    author_b = None
    uploaded_file = None

    if input_method == "Manual":
        author_a = st.text_input("Enter the name of Author 1")
        author_b = st.text_input("Enter the name of Author 2")
    elif input_method == "Random":
        if st.button("Refresh Author List"):
            st.session_state["random_authors"] = random.choices(author_names, k=min(20, len(author_names))) if author_names else []
        random_authors = st.session_state.setdefault(
            "random_authors",
            random.choices(author_names, k=min(20, len(author_names))) if author_names else [],
        )
        col_a, col_b = st.columns(2)
        with col_a:
            author_a = st.selectbox("Select Author 1 (Random)", random_authors)
        with col_b:
            author_b = st.selectbox("Select Author 2 (Random)", random_authors)
    else:
        uploaded_file = st.file_uploader("Upload an author-pair list (.csv or .xlsx)", type=["csv", "xlsx"])

    if st.button("Predict"):
        if not onto_data or not review_year:
            st.error("Please load an ontology and make sure its filename contains a review year.")
        elif input_method == "Batch from File":
            if uploaded_file is None:
                st.warning("Please upload a file for batch prediction.")
            else:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
                if not {"author_A", "author_B"}.issubset(df.columns):
                    st.error("The file must contain 'author_A' and 'author_B' columns.")
                else:
                    model = load_model(model_choice, selected_db, _onto_data=onto_data, _review_year=review_year)
                    predictions = []
                    for _, row in df.iterrows():
                        features = extract_features_for_pair(onto_data, row["author_A"], row["author_B"], review_year)
                        predictions.append(predict_link_prob(features, model) if features is not None and model else None)
                    df["probability"] = predictions
                    st.dataframe(df)
                    st.download_button("Download Prediction Results (.csv)", df.to_csv(index=False).encode("utf-8"), "collaboration_predictions.csv", "text/csv")
        else:
            if not author_a or not author_b:
                st.error("Please enter or select at least two authors.")
            elif author_a == author_b:
                st.error("Please select two different authors.")
            elif author_a not in author_names or author_b not in author_names:
                st.error("One or both authors do not exist in the ontology. Please check the names.")
            else:
                model_type = "GCN" if model_choice == "Graph Convolutional Network (GCN)" else "MLP"
                model = load_model(model_choice, selected_db, _onto_data=onto_data, _review_year=review_year)
                if not model:
                    st.error("Could not load the model. Please check the model file.")
                else:
                    features = extract_features_for_pair(onto_data, author_a, author_b, review_year, model_type=model_type)
                    if features is None:
                        st.error("Could not extract features. Please check the author names and ontology data.")
                    else:
                        if model_type == "GCN":
                            features.author_names_pair = (author_a, author_b)
                        st.subheader("Extracted Features")
                        st.dataframe(features, hide_index=True)
                        probability = predict_link_prob(features, model)

                        if model_type != "GCN" and hasattr(model, "predict_proba"):
                            try:
                                explainer = shap.Explainer(model.predict_proba, features)
                                shap_values = explainer(features)
                                plt.close("all")
                                fig = plt.figure(figsize=(12, 3), dpi=100)
                                output = shap_values[0, :, 1] if len(shap_values.values.shape) > 1 else shap_values[0]
                                shap.plots.force(output, matplotlib=True, show=False)
                                st.subheader("Model Explanation (SHAP)")
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as exc:
                                st.warning(f"Could not create a SHAP plot for **{model_choice}**: {exc}")

                        if probability >= probability_threshold:
                            st.success(f"Authors **{author_a}** and **{author_b}** are likely to collaborate: **{probability * 100:.2f}%**")
                        else:
                            st.warning(f"Authors **{author_a}** and **{author_b}** may not collaborate: **{probability * 100:.2f}%**")

with search_tab:
    st.header("2. Search Information & Collaboration Suggestions 🔍")
    if not onto_data or not author_names:
        st.warning("No ontology or authors are available. Please select an ontology in the sidebar.")
    else:
        search_author_name = st.selectbox("Select an author to search for:", author_names)
        selected_author = next((author for author in authors_full_list if author.name == search_author_name), None)
        if selected_author:
            st.subheader("Author Details")
            st.write(f"**Review Year:** {review_year}")
            st.write(f"**Name:** {getattr(selected_author, 'hasName', search_author_name)}")
            st.write(f"**Affiliation:** {getattr(selected_author, 'hasAffiliation', 'No information available.')}")
            interests = getattr(selected_author, "hasInterestArea", None)
            st.write(f"**Research Interests:** {', '.join(interests) if interests else 'No information available.'}")

        if st.button("Suggest"):
            model = load_model(model_choice, selected_db, _onto_data=onto_data, _review_year=review_year)
            if model:
                candidates = random.choices(author_names, k=min(100, len(author_names)))
                suggestions = suggest_collaborators(search_author_name, model, candidates, onto_data, review_year)
                st.dataframe(suggestions, hide_index=True) if not suggestions.empty else st.write("No suggestions were found.")
            else:
                st.error("Could not load the model. Please check the model file.")

with admin_tab:
    st.header("3. Data and Model Administration 🛠️")
    st.subheader("Ontology Management")
    if st.button("Reload Ontology"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("The cache was cleared. Refresh the page to reload the ontology.")
    st.subheader("Model Management")
    uploaded_model = st.file_uploader("Upload a New Model (.pkl)", type="pkl")
    if uploaded_model:
        st.success("The new model was uploaded successfully.")

with report_tab:
    st.header("4. Export Report 📄")
    st.download_button("Download CSV Data", "author_a,author_b,probability\nJohn,Jane,0.85", "prediction_results.csv", "text/csv")
