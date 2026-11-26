import pandas as pd
import numpy as np
from owlready2 import get_ontology
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
from SPARQLWrapper import SPARQLWrapper
from modules.data_handler import load_authors_list


class NoSuchClassError(Exception):
    pass


def extract_features_for_pair(onto_data, author_A, author_B, review_year, model_type="GCN"):
    """
    Extract features for a specific pair of authors from the ontology.

    Args:
        onto_data (onto): The loaded ontology object.
        author_A (str): Name of the first author.
        author_B (str): Name of the second author.
        review_year (int): Year used to extract historical features.
        model_type (str): Model type ("MLP" or "GCN").

    Returns:
        DataFrame: A DataFrame containing the extracted features,
        or None if no matching authors are found.
    """

    authors = load_authors_list(onto_data)
    author1 = next((a for a in authors if a.name == author_A), None)
    author2 = next((a for a in authors if a.name == author_B), None)
    if not author1 or not author2:
        return None

    aff1 = set(author1.hasAffiliation)
    aff2 = set(author2.hasAffiliation)
    ft1 = 2 * len(aff1 & aff2) / (len(aff1) + len(aff2) + 1e-8)

    interest1 = set(author1.hasInterestArea)
    interest2 = set(author2.hasInterestArea)
    ft2 = 2 * len(interest1 & interest2) / (len(interest1) + len(interest2) + 1e-8)

    papers1 = set(p for p in author1.authored if int(p.hasPublicationYear) < review_year)
    papers2 = set(p for p in author2.authored if int(p.hasPublicationYear) < review_year)
    fttotal = len(papers1 & papers2)

    links1 = set(author1.hasLink) if hasattr(author1, "hasLink") else set()
    links2 = set(author2.hasLink) if hasattr(author2, "hasLink") else set()
    ft3, ft4, ft5 = 0, 0, 0

    common_links = links1 & links2
    if common_links:
        link = next(iter(common_links))
        ft3 = int(link.hasStatus1) if hasattr(link, "hasStatus1") else 0
        ft4 = int(link.hasStatus2) if hasattr(link, "hasStatus2") else 0
        ft5 = int(link.hasStatus3) if hasattr(link, "hasStatus3") else 0

    columns = [
        "hasCommonAffiliation",
        "hasCommonInterest",
        "hasPastStatus",
        "hasPast2",
        "hasPast3",
        "hasPastTotal",
    ]

    if model_type != "GCN":
        features_pair = [ft1, ft2, ft3, ft4, ft5, fttotal]
        return pd.DataFrame([features_pair], columns=columns)

    # For GCN, aggregate features from all co-authorship links
    vec = np.array([ft1, ft2, ft3, ft4, ft5, fttotal])
    author_features = np.zeros((2, 6))
    author_features[0] += vec
    author_features[1] += vec
    author_counts = np.zeros(2)
    author_counts[0] += 1
    author_counts[1] += 1

    links = links1 - common_links
    for link in links:
        if author1 == link.hasAuthor1:
            coAuthor = link.hasAuthor2
        else:
            coAuthor = link.hasAuthor1

        ft1 = 2 * link.hasCommonAffiliation / (1 + link.hasCommonAffiliation)
        ft2 = 2 * link.hasCommonInterest / (1 + link.hasCommonInterest)
        ft3 = int(link.hasStatus1) if hasattr(link, "hasStatus1") else 0
        ft4 = int(link.hasStatus2) if hasattr(link, "hasStatus2") else 0
        ft5 = int(link.hasStatus3) if hasattr(link, "hasStatus3") else 0

        pp1 = set(p for p in author1.authored if int(p.hasPublicationYear) < review_year)
        pp2 = set(p for p in coAuthor.authored if int(p.hasPublicationYear) < review_year)
        fttotal = len(pp1 & pp2)

        author_features[0] += np.array([ft1, ft2, ft3, ft4, ft5, fttotal])
        author_counts[0] += 1

    links = links2 - common_links
    for link in links:
        if author2 == link.hasAuthor1:
            coAuthor = link.hasAuthor2
        else:
            coAuthor = link.hasAuthor1

        ft1 = 2 * link.hasCommonAffiliation / (1 + link.hasCommonAffiliation)
        ft2 = 2 * link.hasCommonInterest / (1 + link.hasCommonInterest)
        ft3 = int(link.hasStatus1) if hasattr(link, "hasStatus1") else 0
        ft4 = int(link.hasStatus2) if hasattr(link, "hasStatus2") else 0
        ft5 = int(link.hasStatus3) if hasattr(link, "hasStatus3") else 0

        pp1 = set(p for p in author2.authored if int(p.hasPublicationYear) < review_year)
        pp2 = set(p for p in coAuthor.authored if int(p.hasPublicationYear) < review_year)
        fttotal = len(pp1 & pp2)

        author_features[1] += np.array([ft1, ft2, ft3, ft4, ft5, fttotal])
        author_counts[1] += 1

    for i in range(2):
        if author_counts[i] > 0:
            author_features[i] /= author_counts[i]

    return pd.DataFrame(
        author_features,
        columns=columns,
        index=[author_A, author_B],
    )
