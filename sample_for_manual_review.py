
import pandas as pd

df = pd.read_csv("moveinsync_reviews.csv")

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating", "review_text"])
df = df.drop_duplicates(subset=["review_text"])

# Oversample low-rated reviews so we can inspect complaints
negative = df[df["rating"] <= 2].sample(
    n=min(75, len(df[df["rating"] <= 2])),
    random_state=42
)

neutral = df[df["rating"] == 3].sample(
    n=min(30, len(df[df["rating"] == 3])),
    random_state=42
)

positive = df[df["rating"] >= 4].sample(
    n=min(45, len(df[df["rating"] >= 4])),
    random_state=42
)

sample = pd.concat(
    [negative, neutral, positive]
).sample(frac=1, random_state=42)

sample = sample[
    ["review_text", "rating", "review_date", "app_version"]
].copy()

sample["manual_category"] = ""
sample["manual_notes"] = ""

sample.to_csv("manual_review_sample.csv", index=False)

print("Sample created successfully!")
print("Reviews selected:", len(sample))
print("Saved: manual_review_sample.csv")