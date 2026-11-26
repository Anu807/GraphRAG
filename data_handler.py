import os
from owlready2 import get_ontology, OwlReadyOntologyParsingError
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
from SPARQLWrapper import SPARQLWrapper


def load_ontology(file_path):
    """Load an ontology file and return the ontology object."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return None

        onto = get_ontology(f"{file_path}")
        print(f"Ontology loaded successfully from: {file_path}")
        return onto
    except OwlReadyOntologyParsingError as exc:
        print(f"Ontology syntax error: {exc}")
        return None
    except Exception as exc:
        print(f"Unknown error while loading ontology: {exc}")
        return None


def load_authors_list(onto_data):
    """Extract the list of authors from the ontology."""
    onto_data.load()
    if onto_data is None:
        print("Could not load the ontology")
        return []

    for cls in onto_data.classes():
        if cls.name == "Author":
            author_class = cls
            break
    else:
        print("The Author class was not found.")
        return []

    authors = author_class.instances()
    print(len(authors))
    if not authors:
        print("No authors were found.")
        return []
    return authors
