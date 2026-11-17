"""Dataset loading and feature preparation for main.ipynb."""

import numpy as np
import pandas as pd

# Issue #4 uses underscores; these are the matching dataset column names.
TIMING_FEATURES = [
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
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
    """Remove flow identifiers, IP addresses, and zero columns; encode Protocol."""
    # New dataframe
    df_stripped = df.set_index("Unnamed: 0").drop(
        columns=["Flow ID", "Source IP", "Destination IP"]
    )  # Exclude flow identifiers and host addresses from the feature set.

    columns_to_drop = [
        x
        for x in df.columns
        if (df[x].dtype == np.float64 or df[x].dtype == np.int64)
        and df[x].min() == 0 == df[x].max()
    ]
    df_stripped = df_stripped.drop(columns=columns_to_drop)

    # Protocol IDs are categories, so keep one numeric indicator per category.
    # Integer indicators use 0 for absent and 1 for present.
    df_stripped = pd.get_dummies(
        df_stripped, columns=["Protocol"], prefix="Protocol", dtype=int
    )

    return df_stripped, columns_to_drop


def plot_flow_iat_histograms(df_timing, *, bins=40):
    """Plot flow inter-arrival-time summaries per class using shared, symlog-spaced bins.

    Normalize each class independently and exclude nonfinite values per feature.
    Work on a copy so plotting never changes the analysis dataframe.
    """
    import matplotlib.pyplot as plt
    from matplotlib.scale import SymmetricalLogScale
    from matplotlib.ticker import PercentFormatter

    features = [name for name in TIMING_FEATURES if name.startswith("Flow IAT ")]
    data = df_timing.loc[:, ["label", *features]].copy()
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    labels = sorted(data["label"].dropna().unique())
    if not labels:
        raise ValueError("No traffic classes are available to plot.")
    transform = SymmetricalLogScale(None, linthresh=1).get_transform()

    for feature in features:
        valid = data.dropna(subset=["label", feature])
        if valid.empty:
            print(f"{feature}: no finite measurements to plot.")
            continue
        low, high = valid[feature].min(), valid[feature].max()
        if low == high:
            low, high = low - 0.5, high + 0.5
        edges = transform.inverted().transform(
            np.linspace(*transform.transform([low, high]), bins + 1)
        )
        # Include the extrema even if the inverse transform rounds slightly.
        edges[0], edges[-1] = low, high
        fig, axes = plt.subplots(
            int(np.ceil(len(labels) / 3)),
            3,
            figsize=(15, 3 * int(np.ceil(len(labels) / 3))),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for ax, label in zip(axes.flat, labels):
            values = valid.loc[valid["label"] == label, feature].to_numpy()
            excluded = int((data["label"] == label).sum()) - len(values)
            if len(values):
                ax.hist(
                    values, bins=edges, weights=np.full(len(values), 1 / len(values))
                )
            ax.set_xscale("symlog", linthresh=1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set_title(f"{label} (valid: {len(values):,}; excluded: {excluded:,})")
            ax.set_xlabel("microseconds, symlog")
            ax.set_ylabel("Share of valid class flows")
            ax.grid(axis="y", alpha=0.3)
        for ax in axes.flat[len(labels) :]:
            ax.set_visible(False)
        fig.suptitle(f"{feature} by traffic class")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()


def plot_directional_iat_scatter(
    df_timing, *, summaries=("Total", "Mean", "Std", "Max", "Min")
):
    """Compare matching forward/backward IAT summaries without changing input data."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summaries = list(summaries)
    if not summaries or any(
        summary not in {"Total", "Mean", "Std", "Max", "Min"} for summary in summaries
    ):
        raise ValueError("Choose IAT summaries from Total, Mean, Std, Max, and Min.")
    features = [
        f"{direction} IAT {summary}"
        for summary in summaries
        for direction in ["Fwd", "Bwd"]
    ]
    data = df_timing.loc[:, ["label", *features]].copy()
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    labels = sorted(data["label"].dropna().unique())
    if not labels:
        raise ValueError("No traffic classes are available to plot.")
    # Use the same class colors in every panel. Mix row order reproducibly so
    # an entire class is not drawn last and placed on top of every other class.
    colors = dict(zip(labels, plt.get_cmap("tab20").colors))
    data = data.sample(frac=1, random_state=0)
    rows = int(np.ceil((len(summaries) + 1) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5.5 * rows), squeeze=False)
    for ax, summary in zip(axes.flat, summaries):
        forward, backward = f"Fwd IAT {summary}", f"Bwd IAT {summary}"
        valid = data.dropna(subset=["label", forward, backward])
        ax.scatter(
            valid[forward],
            valid[backward],
            c=valid["label"].map(colors).tolist(),
            s=8,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        if not valid.empty:
            low = min(valid[forward].min(), valid[backward].min())
            high = max(valid[forward].max(), valid[backward].max())
            if low == high:
                low, high = low - 0.5, high + 0.5
            ax.plot([low, high], [low, high], color="gray", linestyle="--", alpha=0.6)
            # Matching limits make distance from the equality line interpretable.
            ax.set_xlim(low - 0.5, high + 0.5)
            ax.set_ylim(low - 0.5, high + 0.5)
        ax.set_title(
            f"IAT {summary} (valid: {len(valid):,}; excluded: {len(data) - len(valid):,})"
        )
        ax.set_xlabel(f"{forward} (microseconds, symlog)")
        ax.set_ylabel(f"{backward} (microseconds, symlog)")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(summaries) :]:
        ax.set_axis_off()
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=colors[label],
            label=label,
            markersize=6,
        )
        for label in labels
    ]
    axes.flat[-1].legend(handles=handles, title="Traffic class", loc="center")
    fig.suptitle("Forward versus backward inter-arrival times by traffic class")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_directional_iat_correlations(df_timing, *, method="spearman"):
    """Return and plot signed correlations among the ten directional IAT features.

    Use pairwise finite observations without modifying the input dataframe.
    Pass method="pearson" to compare linear relationships instead of ranks.
    """
    import matplotlib.pyplot as plt

    features = [
        name for name in TIMING_FEATURES if name.startswith(("Fwd IAT ", "Bwd IAT "))
    ]
    data = df_timing.loc[:, features].replace([np.inf, -np.inf], np.nan)
    correlations = data.corr(method=method, min_periods=2)
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")
    heatmap = ax.imshow(
        np.ma.masked_invalid(correlations.to_numpy()),
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set_xticks(range(len(features)), labels=features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), labels=features)
    for row in range(len(features)):
        for col in range(len(features)):
            value = correlations.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}" if pd.notna(value) else "N/A",
                ha="center",
                va="center",
                color="white" if pd.notna(value) and abs(value) > 0.7 else "black",
                fontsize=8,
            )
    ax.set_title(f"Forward/backward IAT correlations ({method}, all traffic classes)")
    fig.colorbar(heatmap, ax=ax, label=f"Signed {method} correlation")
    fig.tight_layout()
    plt.show()
    return correlations


def reduce_correlated_features(df, correlations, *, cutoff=0.90):
    """Prune columns in a correlation matrix, preferring Mean summaries.

    Compare each candidate only with already retained columns. This avoids
    dropping a feature because it correlates with another discarded feature.
    Keep undefined correlations and columns outside the matrix unchanged.
    Return a new dataframe and an audit table with signed correlations.
    """
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must be greater than 0 and at most 1.")

    # Python's stable sort preserves matrix order within each priority group.
    candidates = sorted(
        correlations.columns, key=lambda name: "mean" not in name.lower().split()
    )
    kept = []
    removed = []
    for candidate in candidates:
        for retained in kept:
            correlation = correlations.loc[candidate, retained]
            if pd.notna(correlation) and abs(correlation) >= cutoff:
                removed.append((candidate, retained, correlation))
                break
        else:
            kept.append(candidate)

    report = pd.DataFrame(removed, columns=["Removed", "Retained", "Correlation"])
    reduced = df.drop(columns=report["Removed"].tolist()).copy()
    return reduced, report


def plot_backward_packet_rate_boxplots(df_timing):
    """Compare finite Bwd Packets/s by class, displaying rates from -1 upward."""
    import matplotlib.pyplot as plt

    feature = "Bwd Packets/s"
    data = df_timing.loc[:, ["label", feature]].copy()
    data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
    labels = sorted(data["label"].dropna().unique())
    if not labels:
        raise ValueError("No traffic classes are available to plot.")

    values = [data.loc[data["label"] == label, feature].dropna() for label in labels]
    counts = [
        f"{label} (valid: {len(group):,}; "
        f"excluded: {int((data['label'] == label).sum()) - len(group):,})"
        for label, group in zip(labels, values)
    ]
    fig, ax = plt.subplots(figsize=(13, max(5, 0.55 * len(labels))))
    # Compute quartiles on the original rates; scale only the displayed axis.
    ax.boxplot(
        values,
        vert=False,
        whis=1.5,
        showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.25},
        medianprops={"color": "black"},
    )
    ax.set_yticks(range(1, len(labels) + 1), labels=counts)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=-1)
    ax.set_xlabel("Bwd Packets/s (packets per second, symlog)")
    ax.set_ylabel("Traffic class")
    ax.set_title("Backward packet rate by traffic class")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_directional_rate_scatter(df_stripped):
    """Plot directional byte and packet rates in separate figures; return rate data.

    Byte rates use duration in seconds. Invalid durations cannot define byte
    rates, but do not exclude otherwise finite recorded packet rates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = [
        "label",
        "Flow Duration",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packets/s",
        "Bwd Packets/s",
    ]
    data = df_stripped.loc[:, columns].copy()
    numeric = columns[1:]
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
    seconds = data["Flow Duration"].where(data["Flow Duration"] > 0) / 1_000_000
    for direction in ("Fwd", "Bwd"):
        data[f"{direction} Bytes/s"] = (
            data[f"Total Length of {direction} Packets"] / seconds
        )
    rates = data.loc[
        :, ["label", "Fwd Bytes/s", "Bwd Bytes/s", "Fwd Packets/s", "Bwd Packets/s"]
    ].replace([np.inf, -np.inf], np.nan)
    labels = sorted(rates["label"].dropna().unique())
    if not labels:
        raise ValueError("No traffic classes are available to plot.")
    colors = dict(zip(labels, plt.get_cmap("tab20").colors))
    shuffled = rates.sample(frac=1, random_state=0)
    for unit in ("Bytes/s", "Packets/s"):
        forward, backward = f"Fwd {unit}", f"Bwd {unit}"
        valid = shuffled.dropna(subset=["label", forward, backward])
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.scatter(
            valid[forward],
            valid[backward],
            c=valid["label"].map(colors).tolist(),
            s=8,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        if not valid.empty:
            low = min(valid[forward].min(), valid[backward].min())
            high = max(valid[forward].max(), valid[backward].max())
            if low == high:
                low, high = low - 0.5, high + 0.5
            ax.plot([low, high], [low, high], color="gray", linestyle="--", alpha=0.6)
            ax.set_xlim(low - 0.5, high + 0.5)
            ax.set_ylim(low - 0.5, high + 0.5)
        ax.set_xlabel(f"{forward} (symlog)")
        ax.set_ylabel(f"{backward} (symlog)")
        ax.set_title(
            f"Forward versus backward {unit.lower()} by traffic class\n"
            f"Valid pairs: {len(valid):,}; excluded: {len(rates) - len(valid):,}"
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=colors[label],
                label=label,
                markersize=6,
            )
            for label in labels
        ]
        ax.legend(
            handles=handles,
            title="Traffic class",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        plt.show()
    return rates
