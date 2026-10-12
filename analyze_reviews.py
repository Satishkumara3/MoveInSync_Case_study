
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("moveinsync_reviews.csv")

# 2. Check that required columns exist
required = ["review_text", "rating", "review_date"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 3. Clean the data
df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

df = df.dropna(subset=["rating"])
df = df[df["rating"].between(1, 5)]

# 4. Overall metrics
print("\n===== DATASET OVERVIEW =====")
print("Total reviews:", len(df))
print("Unique review texts:", df["review_text"].nunique())
print("Date range:", df["review_date"].min(), "to", df["review_date"].max())
print("Average rating:", round(df["rating"].mean(), 2))

# 5. Rating distribution
print("\n===== RATING DISTRIBUTION =====")
counts = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
distribution = pd.DataFrame({
    "review_count": counts,
    "percentage": (counts / len(df) * 100).round(2)
})
print(distribution)

negative = df["rating"].isin([1, 2]).sum()
print("\n1-2 star reviews:", negative)
print("1-2 star percentage:", round(negative / len(df) * 100, 2), "%")

# 6. Initial rule-based complaint categories
# One review may match more than one category.
categories = {
    "App stability / update": [
        "crash", "crashing", "open", "reinstall", "update",
        "install", "freeze", "hang"
    ],
    "Booking": [
        "book", "booking", "cancel", "cancellation", "reservation"
    ],
    "GPS / tracking / ETA": [
        "gps", "location", "tracking", "track", "eta"
    ],
    "Pickup / trip reliability": [
        "pickup", "pick up", "cab", "driver", "late",
        "delay", "no show", "no-show"
    ],
    "OTP / login / verification": [
        "otp", "login", "log in", "verification", "password"
    ],
    "Notifications": [
        "notification", "alert", "reminder"
    ],
    "UI / usability": [
        "feature", "click", "button", "interface", "annoying",
        "difficult", "confusing", "screen", "step"
    ],
    "Support": [
        "support", "help desk", "customer care", "complaint"
    ]
}

def classify_review(text):
    text = text.lower()
    return [
        category
        for category, keywords in categories.items()
        if any(keyword in text for keyword in keywords)
    ]

df["categories"] = df["review_text"].apply(classify_review)
df["sentiment_group"] = df["rating"].apply(
    lambda rating: "Negative (1-2)" if rating <= 2
    else "Neutral (3)" if rating == 3
    else "Positive (4-5)"
)

# 7. Count reviews by category.
# Counts are not mutually exclusive because a review may have multiple tags.
category_rows = []
for _, row in df.iterrows():
    for category in row["categories"]:
        category_rows.append({
            "category": category,
            "rating": row["rating"],
            "sentiment_group": row["sentiment_group"]
        })

category_df = pd.DataFrame(category_rows)

print("\n===== COMPLAINT / TOPIC COUNTS =====")
if not category_df.empty:
    print(category_df.groupby("category").agg(
        review_count=("rating", "size"),
        average_rating=("rating", "mean"),
        negative_reviews=(
            "sentiment_group",
            lambda s: (s == "Negative (1-2)").sum()
        )
    ).round(2).sort_values("review_count", ascending=False))
else:
    print("No category matches found. Review the keywords.")

# 8. Save outputs
df.to_csv("reviews_cleaned.csv", index=False)

if not category_df.empty:
    category_df.to_csv("category_summary_input.csv", index=False)

print("\nSaved: reviews_cleaned.csv")
print("Analysis completed.")