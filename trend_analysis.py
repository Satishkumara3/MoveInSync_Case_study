
import pandas as pd

# Load reviews
df = pd.read_csv("moveinsync_reviews.csv")

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["review_date"] = pd.to_datetime(
    df["review_date"], errors="coerce"
)

df = df.dropna(subset=["rating", "review_date"])
df = df[df["rating"].between(1, 5)]

df["negative"] = df["rating"] <= 2
df["month"] = df["review_date"].dt.to_period("M").astype(str)

# 1. Monthly rating trend
monthly = df.groupby("month").agg(
    review_count=("rating", "size"),
    average_rating=("rating", "mean"),
    negative_reviews=("negative", "sum")
)

monthly["negative_share_pct"] = (
    monthly["negative_reviews"] / monthly["review_count"] * 100
)

print("\n=== MONTHLY TREND: LAST 18 MONTHS ===")
print(monthly.tail(18).round(2).to_string())

monthly.to_csv("monthly_trend.csv")

# 2. App-version analysis
df["app_version"] = (
    df["app_version"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .replace("", "Unknown")
)

versions = df[df["app_version"] != "Unknown"].groupby(
    "app_version"
).agg(
    review_count=("rating", "size"),
    average_rating=("rating", "mean"),
    negative_reviews=("negative", "sum")
)

versions["negative_share_pct"] = (
    versions["negative_reviews"] / versions["review_count"] * 100
)

# Ignore versions with fewer than 30 reviews
versions = versions[versions["review_count"] >= 30]

versions = versions.sort_values(
    "negative_share_pct", ascending=False
)

print("\n=== APP VERSIONS WITH HIGHEST NEGATIVE SHARE ===")
print(versions.head(15).round(2).to_string())

versions.to_csv("app_version_analysis.csv")

print("\nSaved monthly_trend.csv and app_version_analysis.csv")