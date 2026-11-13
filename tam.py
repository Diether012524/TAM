
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from io import StringIO
from scipy.stats import t


# =====================================
# 1. TAM CONSTRUCTS
# =====================================

constructs = {
    "Perceived Usefulness (PU)": "A. Perceived Usefulness (PU)",
    "Perceived Ease of Use (PEOU)": "B. Perceived Ease of Use (PEOU)",
    "Behavioral Intention (BI)": "C. Behavioral Intention to Use (BI)",
    "Attitude Towards Use (ATU)": "D. Attitude Towards Use (ATU)",
    "Actual Use (AU)": "E. Actual Use (AU)"
}


# =====================================
# 2. VERBAL INTERPRETATION
# =====================================

def interpret_mean(mean):

    if mean >= 3.26:
        return "Strongly Agree"

    elif mean >= 2.51:
        return "Agree"

    elif mean >= 1.76:
        return "Disagree"

    else:
        return "Strongly Disagree"


# =====================================
# 3. VISUALIZATION
# =====================================

def visualize_tam(summary):

    plot_data = summary[
        summary["Construct"] != "Overall TAM"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        plot_data["Construct"],
        plot_data["Mean"],
        edgecolor="black"
    )

    ax.set_title(
        "Technology Acceptance Model (TAM) Results"
    )

    ax.set_ylabel("Mean Score")
    ax.set_xlabel("TAM Constructs")
    ax.set_ylim(0, 4.5)

    ax.axhline(
        3.26,
        linestyle="--",
        linewidth=1.5,
        label="Strongly Agree (3.26)"
    )

    ax.axhline(
        2.51,
        linestyle=":",
        linewidth=1.5,
        label="Agree (2.51)"
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    ax.legend()
    plt.tight_layout()

    filename = "tam_construct_results.png"

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    files.download(filename)


# =====================================
# 4. TAM COMPUTATION
# =====================================

def compute_tam(df):

    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns.astype(str).str.strip()
    )

    # Identify TAM questions
    construct_columns = {}

    for construct, prefix in constructs.items():

        cols = [
            col for col in df.columns
            if col.startswith(prefix)
        ]

        if len(cols) != 4:

            raise ValueError(
                f"{construct} should have 4 questions. "
                f"Found {len(cols)}. "
                f"Check the CSV column names."
            )

        construct_columns[construct] = cols

    # Collect all 20 TAM questions
    all_questions = [
        col
        for cols in construct_columns.values()
        for col in cols
    ]

    # =================================
    # 5. EXTRACT NUMERICAL RATINGS
    # =================================

    ratings = df[all_questions].apply(
        lambda col: pd.to_numeric(
            col.astype(str).str.extract(
                r"^\s*([1-4])(?:\s*-|$)"
            )[0],
            errors="coerce"
        )
    )

    # Check incomplete responses
    incomplete = ratings.isna().any(axis=1)

    print("===== TAM DATA CLEANING =====")

    print(
        f"Total responses loaded: {len(ratings)}"
    )

    print(
        f"Incomplete responses removed: {incomplete.sum()}"
    )

    # Retain complete responses
    valid = ratings.loc[~incomplete].copy()

    if len(valid) == 0:

        raise ValueError(
            "No complete valid responses found."
        )

    if not valid.isin([1, 2, 3, 4]).all().all():

        raise ValueError(
            "Invalid ratings detected. "
            "Only ratings from 1 to 4 are allowed."
        )

    n = len(valid)

    # =================================
    # 6. COMPUTE CONSTRUCT SCORES
    # =================================

    respondent_scores = pd.DataFrame(
        index=valid.index
    )

    results = []

    print("\n===== TAM ANALYSIS RESULTS =====")

    for construct, cols in construct_columns.items():

        # Mean per respondent
        respondent_scores[construct] = (
            valid[cols].mean(axis=1)
        )

        scores = respondent_scores[construct]

        mean_score = scores.mean()
        std_dev = scores.std()

        # 95% Margin of Error
        if n > 1:

            moe = t.ppf(
                0.975,
                n - 1
            ) * (
                std_dev / np.sqrt(n)
            )

        else:
            moe = 0

        interpretation = interpret_mean(
            mean_score
        )

        results.append({
            "Construct": construct,
            "Number of Items": len(cols),
            "Mean": mean_score,
            "Standard Deviation": std_dev,
            "Margin of Error (95%)": moe,
            "Interpretation": interpretation
        })

        print(f"\n===== {construct} =====")

        print(f"Respondents: {n}")
        print(f"Mean: {mean_score:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Margin of Error (95%): ±{moe:.2f}")
        print(f"Interpretation: {interpretation}")

    # =================================
    # 7. OVERALL TAM SCORE
    # =================================

    overall_scores = (
        respondent_scores.mean(axis=1)
    )

    overall_mean = overall_scores.mean()
    overall_std = overall_scores.std()

    if n > 1:

        overall_moe = t.ppf(
            0.975,
            n - 1
        ) * (
            overall_std / np.sqrt(n)
        )

    else:
        overall_moe = 0

    results.append({
        "Construct": "Overall TAM",
        "Number of Items": 20,
        "Mean": overall_mean,
        "Standard Deviation": overall_std,
        "Margin of Error (95%)": overall_moe,
        "Interpretation": interpret_mean(overall_mean)
    })

    print("\n===== OVERALL TAM =====")

    print(f"Respondents: {n}")
    print(f"Overall Mean: {overall_mean:.2f}")
    print(f"Standard Deviation: {overall_std:.2f}")
    print(f"Margin of Error (95%): ±{overall_moe:.2f}")

    print(
        f"Interpretation: {interpret_mean(overall_mean)}"
    )

    # =================================
    # 8. CREATE SUMMARY TABLE
    # =================================

    summary = pd.DataFrame(results)

    print("\n===== TAM SUMMARY TABLE =====")

    print(
        summary.round(2).to_string(index=False)
    )

    # =================================
    # 9. SAVE RESULTS
    # =================================

    summary.to_csv(
        "tam_summary_results.csv",
        index=False
    )

    respondent_scores.to_csv(
        "tam_respondent_scores.csv",
        index=False
    )

    print("\nTAM results successfully saved.")

    files.download(
        "tam_summary_results.csv"
    )

    files.download(
        "tam_respondent_scores.csv"
    )

    # =================================
    # 10. VISUALIZE RESULTS
    # =================================

    visualize_tam(summary)

    return summary, respondent_scores


# =====================================
# 11. UPLOAD CSV FILE IN GOOGLE COLAB
# =====================================

uploaded = files.upload()

csv_file = list(uploaded.keys())[0]

df = pd.read_csv(
    StringIO(
        uploaded[csv_file].decode("utf-8-sig")
    ),
    sep="\t"
)

# Remove unnecessary unnamed columns
df = df.loc[
    :,
    ~df.columns.astype(str).str.contains("^Unnamed")
]

# Run TAM analysis
summary, respondent_scores = compute_tam(df)
