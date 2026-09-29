"""Canonical display names for datasets (literature / package conventions)."""

DISPLAY = {
    "diabetes": "Diabetes",
    "california_housing": "California Housing",
    "wine_quality": "Wine Quality",
    "linnerud": "Linnerud",
    "protein": "Protein",
    "blogfeedback": "BlogFeedback",
    "digits": "Digits",
    "breast_cancer": "Breast Cancer",
    "ionosphere": "Ionosphere",
    "iris": "Iris",
    "wine": "Wine",
    "digits_logistic": "Digits (logistic)",
    "digits_mlp": "Digits (MLP)",
    "breast_cancer_logistic": "Breast Cancer (logistic)",
    "breast_cancer_mlp": "Breast Cancer (MLP)",
    "wine_quality_linear": "Wine Quality (linear)",
    "diabetes_linear": "Diabetes (linear)",
    "california_housing_linear": "California Housing (linear)",
    "Cora": "Cora",
    "CiteSeer": "CiteSeer",
    "PubMed": "PubMed",
    "MNIST": "MNIST",
    "CIFAR-10": "CIFAR-10",
    "CIFAR-100": "CIFAR-100",
}


def pretty(name: str) -> str:
    """Map a dataset key (or mixed-case string) to its display name."""
    if name in DISPLAY:
        return DISPLAY[name]
    key = name.replace(" ", "_").lower()
    return DISPLAY.get(key, name)
