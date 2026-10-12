
import pandas as pd

# Load the manually labelled reviews
df = pd.read_csv("manual_review_sample.csv")

# Check that the required columns exist
required = ["review_text", "rating", "manual_category", "manual_notes"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep only rows that have a manual category
labelled = df[
    df["manual_category"].notna()
    & (df["manual_category"].astype(str).str.strip() != "")
].copy()

if labelled.empty:
    print("No manual categories found. Fill manual_category first.")
else:
    # One review may have multiple categories separated by semicolons
    labelled["manual_category"] = labelled["manual_category"].astype(str)
    labelled["manual_category"] = labelled["manual_category"].str.split(";")
    exploded = labelled.explode("manual_category")
    exploded["manual_category"] = exploded["manual_category"].str.strip()

    # A 1- or 2-star review is counted as negative
    exploded["negative"] = pd.to_numeric(
        exploded["rating"], errors="coerce"
    ).isin([1, 2])

    summary = exploded.groupby("manual_category").agg(
        tagged_reviews=("review_text", "count"),
        negative_reviews=("negative", "sum"),
        average_rating=("rating", "mean")
    ).reset_index()

    summary["negative_share_percent"] = (
        summary["negative_reviews"] / summary["tagged_reviews"] * 100
    ).round(2)

    summary["average_rating"] = summary["average_rating"].round(2)
    summary = summary.sort_values(
        "negative_share_percent", ascending=False
    )

    summary.to_csv("manual_category_summary.csv", index=False)

    print(f"Rows in file: {len(df)}")
    print(f"Manually labelled rows: {len(labelled)}")
    print("\nManual category summary:")
    print(summary.to_string(index=False))
    print("\nSaved results to manual_category_summary.csv")