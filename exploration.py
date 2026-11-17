"""Dataset preparation and PCA helpers for main.ipynb.

Importing this module neither reads data nor fits models. Callers keep the fitted
scaler, PCA, components, and metadata together through PCAAnalysis.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_dataset(path="dataset.csv"):
    """Load flows, trim column names, and name the observed protocol categories."""
    df = pd.read_csv(path)
    df.columns = map(
        lambda x: x.strip(), df.columns
    )  # some columns have space at start

    # Names for the protocol numbers present in this dataset, from the IANA registry.
    # https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
    protocol_names = {0: "HOPOPT", 6: "TCP", 17: "UDP"}
    # Preserve any unrecognized number as its own category instead of losing it.
    df["Protocol"] = (
        df["Protocol"]
        .map(protocol_names)
        .fillna("Unknown_" + df["Protocol"].astype(str))
    )

    return df


def prepare_features(df):
    """Remove redundant identifiers and zero columns; encode Protocol without ordering categories."""
    # New dataframe
    df_stripped = df.set_index("Unnamed: 0").drop(
        columns=["Flow ID"]
    )  # Flow ID is a combination of other attributes

    columns_to_drop = [
        x
        for x in df.columns
        if (df[x].dtype == np.float64 or df[x].dtype == np.int64)
        and df[x].min() == 0 == df[x].max()
    ]
    df_stripped = df_stripped.drop(columns=columns_to_drop)

    # Protocol IDs are categories, so keep one numeric indicator per category.
    # dtype=int makes these columns available to PCA's numeric feature selection.
    df_stripped = pd.get_dummies(
        df_stripped, columns=["Protocol"], prefix="Protocol", dtype=int
    )

    return df_stripped, columns_to_drop


def reduce_correlated_features(df_stripped, *, threshold=0.9):
    """Keep the first quantitative feature from strongly correlated pairs.

    Compare each candidate only with retained features, using absolute Spearman
    correlation. Preserve labels and categorical identifiers, and return a copy
    plus an audit table explaining each removed column.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    features = [
        column
        for column in df_stripped.select_dtypes(include="number").columns
        if column
        not in {"label", "Source Port", "Destination Port", "Protocol", "Inbound"}
        and not column.startswith("Protocol_")
    ]
    correlations = df_stripped[features].corr(method="spearman").abs()
    retained = []
    removed = []
    for feature in features:
        matches = correlations.loc[retained, feature]
        matches = matches[matches > threshold]
        if matches.empty:
            retained.append(feature)
        else:
            representative = matches.idxmax()
            removed.append(
                {
                    "Removed feature": feature,
                    "Retained feature": representative,
                    "Absolute Spearman correlation": matches[representative],
                }
            )
    report = pd.DataFrame(
        removed,
        columns=[
            "Removed feature",
            "Retained feature",
            "Absolute Spearman correlation",
        ],
    )
    return df_stripped.drop(columns=report["Removed feature"].tolist()).copy(), report


@dataclass
class PCAAnalysis:
    """Keep fitted state and aligned data together, without module-level state.

    X_reduced contains only PCs. df_pca also includes restored categories and
    label, so it must not be used directly as classifier input.
    """

    X_pca: pd.DataFrame
    non_quantitative_data: pd.DataFrame
    scaler: StandardScaler
    pca: PCA
    X_components: np.ndarray
    cumulative_variance: np.ndarray
    n_components_95: int
    n_components_99: int
    X_reduced: pd.DataFrame
    df_pca: pd.DataFrame
    y: pd.Series


def fit_pca(df_stripped):
    """Fit standardized quantitative features only; restore categories by row index."""
    # Separate semantic categories as well as nonnumeric columns before scaling.
    # Ports and protocol codes identify categories; their numeric distances are not quantities.
    numeric_features = df_stripped.select_dtypes(include="number").columns
    categorical_numeric_features = [
        column
        for column in numeric_features
        if column in ["Source Port", "Destination Port", "Protocol", "Inbound"]
        or column.startswith("Protocol_")
    ]
    non_quantitative_columns = [
        column
        for column in df_stripped.columns
        if column not in numeric_features or column in categorical_numeric_features
    ]
    non_quantitative_data = df_stripped[non_quantitative_columns].copy()
    X_pca = df_stripped.drop(columns=non_quantitative_columns).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    # Fit every component to select how many retain at least 99% variance.
    pca = PCA(svd_solver="full")
    X_components = pca.fit_transform(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    n_components_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)

    # Reuse fitted scores and retain at least 99% of quantitative variance.
    X_reduced = pd.DataFrame(
        X_components[:, :n_components_99],
        index=X_pca.index,
        columns=[f"PC{i}" for i in range(1, n_components_99 + 1)],
    )

    # Restore categories and metadata using the original row index, without transforming them.
    df_pca = pd.concat([X_reduced, non_quantitative_data], axis=1)
    y = df_pca["label"].copy()

    return PCAAnalysis(
        X_pca,
        non_quantitative_data,
        scaler,
        pca,
        X_components,
        cumulative_variance,
        n_components_95,
        n_components_99,
        X_reduced,
        df_pca,
        y,
    )
