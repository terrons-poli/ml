"""Data preparation and plotting helpers for main.ipynb.

Functions take explicit inputs and clean plotting copies rather than mutating the
analysis dataframe. Importing this module neither reads data nor fits PCA.
The notebook controls which analyses run and keeps their explanations and outputs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import SymmetricalLogScale
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display

FEATURES = [
    "Flow Packets/s",
    "Flow Duration",
    "Packet Length Mean",
    "Packet Length Std",
]


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


def plot_class_counts(df_stripped):
    """Plot recorded flow counts per class, rather than attack-incident counts."""
    dns_counts = df_stripped.groupby("label").size().sort_values(ascending=False)

    sns.barplot(
        x=dns_counts.values,
        y=dns_counts.index,
        hue=["Legit" if x == "benign" else "Attack" for x in dns_counts.index],
        palette={"Legit": "#6868ff", "Attack": "#fd6868"},
        orient="h",
    )
    plt.title("Number of flows per label")
    plt.xlabel("# of flows")
    plt.show()


def plot_protocol_percentages(df):
    """Compare protocol composition within each label using pre-encoding names.

    Normalize rows, not columns, so every class totals 100% regardless of size.
    Return percentages for reuse without modifying the input dataframe.
    """
    protocol_data = (
        df.rename(columns=lambda name: name.strip())
        .loc[:, ["label", "Protocol"]]
        .dropna()
        .copy()
    )
    percentages = pd.crosstab(
        protocol_data["label"], protocol_data["Protocol"], normalize="index"
    ).mul(100)

    ax = percentages.plot.barh(
        stacked=True,
        figsize=(12, 8),
        color=sns.color_palette("colorblind", n_colors=len(percentages.columns)),
        width=0.8,
    )
    ax.invert_yaxis()
    ax.set_title("Protocol percentages within each traffic class")
    ax.set_xlabel("Percentage of flows within class")
    ax.set_ylabel("Traffic class (label)")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend(title="Protocol", bbox_to_anchor=(1.02, 1), loc="upper left")
    # Small shares remain visible but labels would overlap inside narrow segments.
    for container in ax.containers:
        labels = [
            f"{bar.get_width():.1f}%" if bar.get_width() >= 5 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=9)
    plt.tight_layout()
    plt.show()
    return percentages


def _distribution_data(df_stripped, features=None):
    """Build a finite plotting copy; preserve the original dataframe."""
    box_features = FEATURES if features is None else list(features)
    # Strip column-name whitespace locally without changing the analysis dataframe.
    box_data = (
        df_stripped.rename(columns=lambda name: name.strip())
        .loc[:, ["label", *box_features]]
        .copy()
    )
    box_data[box_features] = box_data[box_features].replace([np.inf, -np.inf], np.nan)
    box_label_order = sorted(box_data["label"].dropna().unique())

    return box_data, box_label_order


def plot_feature_boxplots(df_stripped, features=None):
    """Compare class medians and spreads; use symlog for strongly skewed features."""
    box_features = FEATURES if features is None else list(features)
    box_data, box_label_order = _distribution_data(df_stripped, box_features)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    for ax, feature in zip(axes.flat, box_features):
        # Drop invalid rows per feature so missing values in other features do not interfere.
        feature_data = box_data.dropna(subset=["label", feature])
        sns.boxplot(
            data=feature_data,
            x="label",
            y=feature,
            order=box_label_order,
            color="#6868ff",
            flierprops={"markersize": 2, "alpha": 0.3},
            ax=ax,
        )
        # A skewness magnitude above 2 indicates a strong asymmetric tail.
        if abs(feature_data[feature].skew()) > 2:
            ax.set_yscale("symlog", linthresh=1)
            scale_label = "symlog scale; linear from 0 to 1"
        else:
            scale_label = "linear scale"
        # Leave a small margin below zero, bounded at -10^0.
        ax.set_ylim(bottom=-1)
        ax.set_title(f"{feature} by traffic class")
        ax.set_xlabel("Traffic class (label)")
        ax.set_ylabel(f"{feature} ({scale_label})")
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_active_features(df_stripped):
    """Compare active-period duration summaries with the existing four-panel style.

    Reuse the boxplot helper so filtering, class order, and scale selection remain
    consistent with the earlier feature comparisons. Only plotting copies change.
    """
    plot_feature_boxplots(
        df_stripped, features=["Active Mean", "Active Std", "Active Max", "Active Min"]
    )

    # Normalize within each label so classes with different sizes are comparable.
    active_data = df_stripped.rename(columns=lambda name: name.strip())
    active_mean = active_data["Active Mean"].replace([np.inf, -np.inf], np.nan)
    positive_percent = (
        active_mean.gt(0)
        .groupby(active_data["label"])
        .mean()
        .mul(100)
        .sort_values(ascending=False, kind="stable")
    )
    print("Flows with Active Mean > 0 (% of all flows within each label):")
    print(positive_percent.to_string(float_format=lambda value: f"{value:.2f}%"))
    missing_count = int(active_mean.isna().sum())
    if missing_count:
        print(
            f"Missing/nonfinite Active Mean: {missing_count:,}; included in class totals, not counted as positive."
        )


def plot_feature_histograms(df_stripped):
    """Compare per-class proportions using shared bins and zero-safe scales."""
    box_features = FEATURES
    box_data, box_label_order = _distribution_data(df_stripped)
    for feature in box_features:
        histogram_data = box_data.dropna(subset=["label", feature])
        use_symlog = abs(histogram_data[feature].skew()) > 2
        values = histogram_data[feature].to_numpy()

        # Space bins evenly on the displayed scale, using the same edges for every class.
        if use_symlog:
            scale = SymmetricalLogScale(None, base=10, linthresh=1)
            transform = scale.get_transform()
            limits = transform.transform(np.array([values.min(), values.max()]))
            bin_edges = transform.inverted().transform(np.linspace(*limits, 41))
            # Avoid round-off excluding the smallest or largest observation.
            bin_edges[0], bin_edges[-1] = values.min(), values.max()
            scale_label = "symlog scale; linear between -1 and 1"
        else:
            bin_edges = np.histogram_bin_edges(values, bins=40)
            scale_label = "linear scale"

        ncols = 4
        nrows = int(np.ceil(len(box_label_order) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(20, 3.5 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for ax, label in zip(axes.flat, box_label_order):
            class_values = histogram_data.loc[
                histogram_data["label"] == label, feature
            ].to_numpy()
            # Normalize counts directly; Matplotlib accepts bins in original units.
            weights = (
                np.full(len(class_values), 1 / len(class_values))
                if len(class_values)
                else None
            )
            ax.hist(
                class_values,
                bins=bin_edges,
                weights=weights,
                color="#6868ff",
                alpha=0.7,
            )
            if use_symlog:
                ax.set_xscale("symlog", linthresh=1)
            ax.set_title(label)
            ax.set_xlabel(f"{feature} ({scale_label})")
            ax.set_ylabel("Fraction of class")
            ax.tick_params(axis="x", labelrotation=45)
            ax.grid(axis="y", alpha=0.3)

        for ax in list(axes.flat)[len(box_label_order) :]:
            ax.set_visible(False)
        fig.suptitle(f"{feature} distribution by traffic class")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()


def plot_byte_rates(df_stripped):
    """Calculate directional rates locally and plot them with class colors."""
    byte_rate_data = (
        df_stripped.rename(columns=lambda name: name.strip())
        .loc[
            :,
            [
                "label",
                "Flow Duration",
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
            ],
        ]
        .copy()
    )
    byte_rate_data = byte_rate_data.replace([np.inf, -np.inf], np.nan)
    # Nonpositive durations cannot define a valid bytes-per-second rate.
    duration_seconds = (
        byte_rate_data["Flow Duration"].where(byte_rate_data["Flow Duration"] > 0)
        / 1_000_000
    )
    byte_rate_data["Fwd Bytes/s"] = (
        byte_rate_data["Total Length of Fwd Packets"] / duration_seconds
    )
    byte_rate_data["Bwd Bytes/s"] = (
        byte_rate_data["Total Length of Bwd Packets"] / duration_seconds
    )
    byte_rate_data = byte_rate_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["label", "Fwd Bytes/s", "Bwd Bytes/s"]
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=byte_rate_data,
        x="Fwd Bytes/s",
        y="Bwd Bytes/s",
        hue="label",
        palette="husl",
        alpha=0.4,
        s=15,
        linewidth=0,
        ax=ax,
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    # Byte rates are nonnegative; remove the empty negative axis margins.
    ax.set_xlim(left=-1)
    ax.set_ylim(bottom=-1)
    ax.set_title("Forward versus backward flow byte rates by traffic class")
    ax.set_xlabel("Forward bytes/s (symlog scale; linear from 0 to 1)")
    ax.set_ylabel("Backward bytes/s (symlog scale; linear from 0 to 1)")
    ax.legend(title="Traffic class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _port_data(df_stripped):
    """Keep valid integer ports in a local copy, including zero."""
    port_features = ["Source Port", "Destination Port"]
    port_data = (
        df_stripped.rename(columns=lambda name: name.strip())
        .loc[:, ["label", *port_features]]
        .copy()
    )
    port_data[port_features] = port_data[port_features].replace(
        [np.inf, -np.inf], np.nan
    )
    # Valid port identifiers are integers from 0 through 65535, including zero.
    for feature in port_features:
        valid_ports = port_data[feature].between(0, 65535) & (
            port_data[feature] % 1 == 0
        )
        port_data[feature] = port_data[feature].where(valid_ports)

    return port_data


def plot_port_ecdfs(df_stripped):
    """Compare separately normalized port ECDFs for every class."""
    port_features = ["Source Port", "Destination Port"]
    port_data = _port_data(df_stripped)
    port_label_order = sorted(port_data["label"].dropna().unique())
    port_palette = dict(
        zip(port_label_order, sns.color_palette("husl", n_colors=len(port_label_order)))
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    for ax, feature in zip(axes, port_features):
        feature_data = port_data.dropna(subset=["label", feature])
        sns.ecdfplot(
            data=feature_data,
            x=feature,
            hue="label",
            hue_order=port_label_order,
            palette=port_palette,
            stat="proportion",
            linewidth=1.8,
            legend=(ax is axes[1]),
            ax=ax,
        )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, 65535)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{feature} ECDF by traffic class")
        ax.set_xlabel(f"{feature} (symlog scale; linear from 0 to 1)")
        ax.set_ylabel("Fraction of class at or below port")
        ax.grid(alpha=0.3)

    sns.move_legend(
        axes[1], "upper left", bbox_to_anchor=(1.02, 1), title="Traffic class"
    )
    plt.tight_layout()
    plt.show()


def show_frequent_ports(df_stripped):
    """Display five independently ranked ports per direction and class; return numeric data."""
    port_data = _port_data(df_stripped)
    top_port_tables = []
    for feature, direction in [
        ("Source Port", "Source"),
        ("Destination Port", "Destination"),
    ]:
        counts = (
            port_data.dropna(subset=["label", feature])
            .groupby(["label", feature])
            .size()
            .rename("Count")
            .reset_index()
        )
        counts["Percent"] = (
            100 * counts["Count"] / counts.groupby("label")["Count"].transform("sum")
        )
        counts = counts.sort_values(
            ["label", "Count", feature], ascending=[True, False, True]
        )
        counts["Rank"] = counts.groupby("label").cumcount() + 1
        counts = counts[counts["Rank"] <= 5].copy()
        counts[feature] = counts[feature].astype("Int64")
        top_port_tables.append(
            counts.rename(
                columns={
                    feature: f"{direction} port",
                    "Count": f"{direction} count",
                    "Percent": f"{direction} percent",
                }
            )
        )

    port_frequency_table = (
        top_port_tables[0]
        .merge(top_port_tables[1], on=["label", "Rank"], how="outer")
        .sort_values(["label", "Rank"])
        .reset_index(drop=True)
    )

    # Format a display copy, preserving numeric percentages in the reusable table.
    port_frequency_display = port_frequency_table.copy()
    for column in ["Source percent", "Destination percent"]:
        port_frequency_display[column] = port_frequency_display[column].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else "N/A"
        )
    # Show every class rather than allowing pandas to truncate the middle rows.
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(port_frequency_display)

    return port_frequency_table


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


def plot_correlations(df_stripped, *, figsize=(24, 20)):
    """Show absolute Spearman correlations; figsize sets width and height in inches."""
    corr_matrix = df_stripped.corr(numeric_only=True, method="spearman").abs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap="Blues",
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    plt.show()


@dataclass
class PCAAnalysis:
    """Keep fitte state and aligned data together, without module-level state.

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

    @property
    def loadings(self):
        """Scale coefficients by component standard deviation in exact input order."""
        return pd.DataFrame(
            self.pca.components_.T * np.sqrt(self.pca.explained_variance_),
            index=self.X_pca.columns,
            columns=[f"PC{i}" for i in range(1, self.pca.n_components_ + 1)],
        )


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

    # Fit every component so the cumulative curve covers all available variance.
    pca = PCA(svd_solver="full")
    X_components = pca.fit_transform(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    component_counts = np.arange(1, len(cumulative_variance) + 1)
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


def plot_explained_variance(analysis):
    """Show cumulative quantitative variance with 95% and 99% thresholds."""
    pca = analysis.pca
    X_pca = analysis.X_pca
    cumulative_variance = analysis.cumulative_variance
    component_counts = np.arange(1, len(cumulative_variance) + 1)
    n_components_95 = analysis.n_components_95
    n_components_99 = analysis.n_components_99
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(component_counts, cumulative_variance, marker="o", markersize=3)
    ax.axhline(0.95, color="red", linestyle="--", label="95% explained variance")
    ax.axhline(0.99, color="green", linestyle="--", label="99% explained variance")
    ax.axvline(
        n_components_95,
        color="gray",
        linestyle=":",
        label=f"{n_components_95} components retain at least 95%",
    )
    ax.axvline(
        n_components_99,
        color="gray",
        linestyle=":",
        label=f"{n_components_99} components retain at least 99%",
    )
    ax.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance ratio",
        title="PCA cumulative explained variance",
        xlim=(1, len(cumulative_variance)),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Quantitative features entering PCA: {X_pca.shape[1]}")
    print(f"Components needed for at least 95% variance: {n_components_95}")
    print(f"Components needed for at least 99% variance: {n_components_99}")


def show_pca_dataframe(analysis):
    """Summarize reduced measurements and restored metadata, then display sample rows."""
    print(f"Component-only shape: {analysis.X_reduced.shape}")
    print(
        f"Appended nonquantitative columns: {list(analysis.non_quantitative_data.columns)}"
    )
    print(f"Combined dataframe shape: {analysis.df_pca.shape}")
    print(
        f"Retained quantitative variance: {analysis.cumulative_variance[analysis.n_components_99 - 1]:.2%}"
    )
    display(analysis.df_pca.head())


def plot_pca_loadings(analysis):
    """Plot the ten largest signed scaled loadings of PC1 through PC3."""
    pca = analysis.pca
    pca_loadings = analysis.loadings
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for component_index, ax in enumerate(axes):
        component = f"PC{component_index + 1}"
        loadings = pca_loadings[component]
        top_features = loadings.abs().nlargest(10).index
        top_loadings = loadings.loc[top_features].iloc[::-1]
        colors = ["#6868ff" if value >= 0 else "#fd6868" for value in top_loadings]
        ax.barh(top_loadings.index, top_loadings.values, color=colors)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_title(
            f"{component}: top 10 loadings\n"
            f"{pca.explained_variance_ratio_[component_index]:.2%} explained variance"
        )
        ax.set_xlabel("Signed feature loading")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_pca_loadings_heatmap(analysis, *, signed=True):
    """Show top-ten features from PC1–PC5; signed=False displays absolute loadings.

    Signed values use a zero-centered diverging palette to show direction.
    Absolute values use the same palette on a nonnegative scale to compare strength.
    Feature selection always uses magnitude, and stored loadings stay unchanged.
    """
    pca = analysis.pca
    pca_loadings = analysis.loadings
    heatmap_components = [f"PC{i}" for i in range(1, 6)]
    # Keep the union in first-seen rank order, without duplicates.
    heatmap_features = list(
        dict.fromkeys(
            feature
            for component in heatmap_components
            for feature in pca_loadings[component].abs().nlargest(10).index
        )
    )
    heatmap_loadings = pca_loadings.loc[heatmap_features, heatmap_components]
    if not signed:
        heatmap_loadings = heatmap_loadings.abs()
    heatmap_labels = [
        f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.2%})" for i in range(5)
    ]
    color_limit = heatmap_loadings.abs().to_numpy().max()

    fig, ax = plt.subplots(figsize=(11, max(8, 0.32 * len(heatmap_features))))
    sns.heatmap(
        heatmap_loadings,
        cmap="RdBu_r",
        center=0 if signed else None,
        vmin=-color_limit if signed else 0,
        vmax=color_limit,
        xticklabels=heatmap_labels,
        yticklabels=True,
        cbar_kws={
            "label": "Signed feature loading" if signed else "Absolute feature loading"
        },
        ax=ax,
    )
    ax.set_title("PCA loadings: union of the top 10 features per PC")
    ax.set_xlabel("Principal component (explained variance)")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(analysis, *, group="all", zoom=False, xlim=None):
    """Plot existing scores; optional group filtering never refits PCA. Return PC1 limits."""
    all_scores = analysis.X_reduced[["PC1", "PC2"]].copy()
    all_scores["label"] = analysis.y
    class_order = sorted(all_scores["label"].dropna().unique())
    palette = dict(
        zip(class_order, sns.color_palette("husl", n_colors=len(class_order)))
    )
    if group == "all":
        scores = all_scores
        title = "PC1 versus PC2 by traffic class"
    elif group == "benign":
        scores = all_scores.loc[all_scores["label"] == "benign"]
        title = "Benign traffic: PC1 versus PC2"
    elif group == "ddos":
        scores = all_scores.loc[all_scores["label"].str.startswith("ddos_", na=False)]
        title = "DDoS traffic: PC1 versus PC2"
    else:
        raise ValueError("group must be 'all', 'benign', or 'ddos'")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=scores,
        x="PC1",
        y="PC2",
        hue="label",
        palette=palette,
        alpha=0.4,
        s=15,
        linewidth=0,
        ax=ax,
    )
    # Derive the full limits from all flows, even when displaying only one group.
    # Matplotlib's default scatter margins add 5% of the data range on either side.
    if xlim is None:
        low, high = all_scores["PC1"].min(), all_scores["PC1"].max()
        margin = 0.05 * (high - low)
        xlim = (low - margin, high + margin)
    ax.set_xlim(xlim)
    if zoom:
        low, high = all_scores["PC2"].quantile([0.01, 0.99])
        ax.set_ylim(low, high)
        title += ": zoomed to PC2's 1st–99th percentiles"
    ax.set_title(title)
    ax.set_xlabel(
        f"PC1 ({analysis.pca.explained_variance_ratio_[0]:.2%} explained variance)"
    )
    ax.set_ylabel(
        f"PC2 ({analysis.pca.explained_variance_ratio_[1]:.2%} explained variance)"
    )
    ax.legend(title="Traffic class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    if zoom:
        below = int((scores["PC2"] < low).sum())
        above = int((scores["PC2"] > high).sum())
        total = len(scores)
        print(f"Displayed PC2 range: [{low:.6g}, {high:.6g}]")
        print(f"{group.capitalize()} flows: {total:,}")
        print(f"Below range: {below:,} ({below / total:.2%})")
        print(f"Above range: {above:,} ({above / total:.2%})")
        print(f"Total outside view: {below + above:,} ({(below + above) / total:.2%})")
    return ax.get_xlim()


def plot_pc1_histogram(analysis):
    """Compare PC1 distributions with shared bins and independent density normalization."""
    pca = analysis.pca
    pc1_histogram_data = analysis.X_reduced[["PC1"]].copy()
    pc1_histogram_data["label"] = analysis.y
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(
        data=pc1_histogram_data,
        x="PC1",
        hue="label",
        hue_order=sorted(pc1_histogram_data["label"].dropna().unique()),
        palette="husl",
        bins=60,
        stat="density",
        common_norm=False,
        common_bins=True,
        multiple="layer",
        element="step",
        fill=False,
        linewidth=1.5,
        ax=ax,
    )
    ax.set_title("PC1 distribution by traffic class")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} explained variance)")
    ax.set_ylabel("Probability density within each class")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="Traffic class")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
