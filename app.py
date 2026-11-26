import os
import random
import re

import pandas as pd
import streamlit as st

from modules.data_handler import load_ontology, load_authors_list
from modules.feature_extraction import extract_features_for_pair
from modules.model_handler import load_model, predict_link_prob, suggest_collaborators

st.set_page_config(layout="wide", page_title="Academic GraphRAG")

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 30% !important;
    transition: width 0.3s ease-in-out;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ontology...")
def load_cached_ontology(path):
    """Load an ontology and cache it."""
    if not path:
        return None
    try:
        return load_ontology(path)
    except Exception as exc:
        st.error(f"Could not load the ontology: {exc}. Please check the file.")
        return None


@st.cache_data(show_spinner="Loading the author list...")
def get_cached_author_list(ontology):
    """Extract the author list from the ontology."""
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
    databases = sorted(
        name for name in os.listdir("data")
        if os.path.isdir(os.path.join("data", name))
    )
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
        owl_files = sorted(
            name for name in os.listdir(ontologies_dir)
            if name.lower().endswith(".owl")
        )
        if not owl_files:
            st.sidebar.warning(f"No ontology files were found in '{ontologies_dir}'.")

        selected_ontology = st.sidebar.selectbox(
            "Please select an ontology file:",
            owl_files,
        )
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
    "Link Prediction",
    "Search and Suggestions",
    "Administration",
    "Export Report",
])

with link_tab:
    st.header("1. Link Prediction 🤝")

    input_method = st.radio(
        "Choose an input method",
        ("Manual", "Random", "Batch from File"),
    )

    author_a = None
    author_b = None
    uploaded_file = None

    if input_method == "Manual":
        st.subheader("Manual Input")
        author_a = st.text_input("Enter the name of Author 1")
        author_b = st.text_input("Enter the name of Author 2")

    elif input_method == "Random":
        st.subheader("Random Author Selection")
        if "random_authors" not in st.session_state:
            st.session_state.random_authors = (
                random.choices(author_names, k=min(20, len(author_names)))
                if author_names else []
            )

        if st.button("Refresh Author List"):
            st.session_state.random_authors = (
                random.choices(author_names, k=min(20, len(author_names)))
                if author_names else []
            )

        random_authors = st.session_state.random_authors
        col_a, col_b = st.columns(2)
        with col_a:
            author_a = st.selectbox("Select Author 1 (Random)", random_authors)
        with col_b:
            author_b = st.selectbox("Select Author 2 (Random)", random_authors)

    else:
        st.subheader("Batch Prediction from a File")
        uploaded_file = st.file_uploader(
            "Upload an author-pair list (.csv or .xlsx)",
            type=["csv", "xlsx"],
        )

    if st.button("Predict"):
        if not onto_data or not review_year:
            st.error("Please load an ontology and make sure it contains a review year.")

        elif input_method == "Batch from File":
            if uploaded_file is None:
                st.warning("Please upload a file for batch prediction.")
            else:
                try:
                    if uploaded_file.name.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    if not {"author_A", "author_B"}.issubset(df.columns):
                        st.error("The file must contain 'author_A' and 'author_B' columns.")
                    else:
                        model = load_model(
                            model_choice,
                            selected_db,
                            _onto_data=onto_data,
                            _review_year=review_year,
                        )
                        if not model:
                            st.error("Could not load the model. Please check the model file.")
                        else:
                            predictions = []
                            with st.spinner("Processing batch predictions..."):
                                for _, row in df.iterrows():
                                    features = extract_features_for_pair(
                                        onto_data,
                                        row["author_A"],
                                        row["author_B"],
                                        review_year,
                                    )
                                    probability = (
                                        predict_link_prob(features, model)
                                        if features is not None else None
                                    )
                                    predictions.append(probability)

                            df["probability"] = predictions
                            st.subheader("Batch Prediction Results")
                            st.dataframe(df)
                            st.download_button(
                                "Download Prediction Results (.csv)",
                                df.to_csv(index=False).encode("utf-8"),
                                "collaboration_predictions.csv",
                                "text/csv",
                            )
                except Exception as exc:
                    st.error(f"Could not read the file: {exc}")

        else:
            if not author_a or not author_b:
                st.error("Please enter or select at least two authors.")
            elif author_a == author_b:
                st.error("Please select two different authors.")
            elif author_a not in author_names or author_b not in author_names:
                st.error("One or both authors do not exist in the ontology. Please check the names.")
            else:
                if model_choice == "Graph Convolutional Network (GCN)":
                    model_type = "GCN"
                    model = load_model(
                        model_choice,
                        selected_db,
                        _onto_data=onto_data,
                        _review_year=review_year,
                    )
                else:
                    model_type = "MLP"
                    model = load_model(model_choice, selected_db)

                if not model:
                    st.error("Could not load the model. Please check the model file.")
                else:
                    features = extract_features_for_pair(
                        onto_data,
                        author_a,
                        author_b,
                        review_year,
                        model_type=model_type,
                    )
                    if features is None:
                        st.error("Could not extract features. Please check the author names and ontology data.")
                    else:
                        if model_type == "GCN":
                            features.author_names_pair = (author_a, author_b)

                        st.subheader("Extracted Features")
                        st.dataframe(features, hide_index=True)
                        probability = predict_link_prob(features, model)

                        common_affiliation = float(features.iloc[0].get("hasCommonAffiliation", 0))
                        common_interest = float(features.iloc[0].get("hasCommonInterest", 0))
                        past_1 = int(features.iloc[0].get("hasPastStatus", 0))
                        past_2 = int(features.iloc[0].get("hasPast2", 0))
                        past_3 = int(features.iloc[0].get("hasPast3", 0))
                        past_total = int(features.iloc[0].get("hasPastTotal", 0))

                        explanation = f"**{author_a}** and **{author_b}** "
                        explanation += (
                            f"share **{common_affiliation * 100:.2f}%** affiliation similarity; "
                            if common_affiliation > 0 else "have no shared affiliation; "
                        )
                        explanation += (
                            f"share **{common_interest * 100:.2f}%** research-interest similarity; "
                            if common_interest > 0 else "have no shared research interests; "
                        )
                        if past_1:
                            explanation += f"co-authored **{past_1}** publication(s) in the previous year; "
                        if past_2:
                            explanation += f"co-authored **{past_2}** publication(s) in the previous two years; "
                        if past_3:
                            explanation += f"co-authored **{past_3}** publication(s) in the previous three years; "
                        explanation += f"and co-authored **{past_total}** publication(s) in total."

                        if probability >= probability_threshold:
                            explanation += f" The probability is **{probability * 100:.2f}%**, above the selected threshold."
                            st.success(f"Authors **{author_a}** and **{author_b}** are likely to collaborate.")
                        else:
                            explanation += f" The probability is **{probability * 100:.2f}%**, below the selected threshold."
                            st.warning(f"Authors **{author_a}** and **{author_b}** may not collaborate.")
                        st.info(f"**Explanation:** {explanation}")

with search_tab:
    st.header("2. Search Information & Collaboration Suggestions 🔍")
    if not onto_data:
        st.warning("No ontology is loaded. Please select an ontology in the sidebar.")
    elif not author_names:
        st.warning("No authors were found in the selected ontology.")
    else:
        search_author_name = st.selectbox("Select an author to search for:", author_names)
        selected_author = next(
            (author for author in authors_full_list if author.name == search_author_name),
            None,
        )

        if selected_author:
            st.subheader("Author Details")
            st.write(f"**Review Year:** {review_year}")
            st.write(f"**Name:** {selected_author.hasName}")

            affiliation = getattr(selected_author, "hasAffiliation", None)
            st.write(f"**Affiliation:** {affiliation if affiliation else 'No information available.'}")

            interests = getattr(selected_author, "hasInterestArea", None)
            st.write(
                f"**Research Interests:** {', '.join(interests) if interests else 'No information available.'}"
            )

        st.subheader("Collaboration Suggestions")
        if st.button("Suggest"):
            if not review_year:
                st.error("Please ensure that the ontology corresponds to a review year.")
            else:
                if model_choice == "Graph Convolutional Network (GCN)":
                    suggestion_model = load_model(
                        model_choice,
                        selected_db,
                        _onto_data=onto_data,
                        _review_year=review_year,
                    )
                else:
                    suggestion_model = load_model(model_choice, selected_db)

                if suggestion_model:
                    candidates = random.choices(author_names, k=min(100, len(author_names)))
                    suggestions = suggest_collaborators(
                        search_author_name,
                        suggestion_model,
                        candidates,
                        onto_data,
                        review_year,
                    )
                    if suggestions.empty:
                        st.write("No suggestions were found.")
                    else:
                        st.dataframe(suggestions, hide_index=True)
                else:
                    st.error("Could not load the model. Please check the model file.")

with admin_tab:
    st.header("3. Data and Model Administration 🛠️")
    st.info("This area is intended for administrators.")

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
    st.subheader("Export Prediction Results")
    st.download_button(
        "Download CSV Data",
        "author_a,author_b,probability\nJohn,Jane,0.85",
        "prediction_results.csv",
        "text/csv",
    )
