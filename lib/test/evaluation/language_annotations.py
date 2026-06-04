import csv
import os


def _default_annotation_path():
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data_specs", "language_descriptions.csv"))


def load_language_annotations(path=None):
    path = path or os.environ.get("DUTRACK_LANGUAGE_ANNOTATIONS", _default_annotation_path())
    annotations = {}
    if not path or not os.path.isfile(path):
        return annotations
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            sequence = str(row.get("sequence", "")).strip()
            description = str(row.get("description", "")).strip()
            if dataset and sequence and description:
                annotations[(dataset, sequence)] = " ".join(description.split())
    return annotations


def lookup_language_description(dataset_name, sequence_name, default=""):
    annotations = load_language_annotations()
    return annotations.get((dataset_name, sequence_name), default)
