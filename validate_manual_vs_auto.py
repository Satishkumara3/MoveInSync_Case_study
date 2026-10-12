
import pandas as pd
import ast
import re

manual = pd.read_csv("manual_review_sample.csv")
auto = pd.read_csv("tagged_reviews.csv")

# Standardize review text for matching
def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()

for df in [manual, auto]:
    df["match_text"] = df["review_text"].apply(clean_text)
    df["match_rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["match_date"] = pd.to_datetime(
        df["review_date"], errors="coerce"
    )

# Convert category values into sets
def parse_categories(value):
    if pd.isna(value):
        return set()

    if isinstance(value, (list, tuple, set)):
        return {str(x).strip() for x in value if str(x).strip()}

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return {str(x).strip() for x in parsed if str(x).strip()}
    except (ValueError, SyntaxError):
        pass

    return {text} if text else set()

# Map different category names to the same standard label
def standardize(category):
    c = category.lower().strip()

    if "gps" in c or "tracking" in c or "eta" in c:
        return "GPS / tracking / ETA"
    if "stability" in c or "update" in c or "crash" in c:
        return "App stability / update"
    if "pickup" in c or "trip reliability" in c:
        return "Pickup / trip reliability"
    if "otp" in c or "login" in c or "verification" in c or "registration" in c:
        return "OTP / login / registration"
    if "booking" in c or "scheduling" in c:
        return "Booking / scheduling"
    if "notification" in c:
        return "Notifications"
    if "ui" in c or "usability" in c:
        return "UI / usability"
    if "support" in c or "communication" in c:
        return "Support / communication"

    return None

manual["manual_set"] = manual["manual_category"].apply(
    parse_categories
).apply(lambda values: {
    standardize(x) for x in values if standardize(x)
})

auto["auto_set"] = auto["categories"].apply(
    parse_categories
).apply(lambda values: {
    standardize(x) for x in values if standardize(x)
})

# Match reviews by text, rating and date
keys = ["match_text", "match_rating", "match_date"]

# Avoid ambiguous many-to-many matches
auto_unique = auto.drop_duplicates(keys, keep="first")

merged = manual.merge(
    auto_unique[keys + ["auto_set"]],
    on=keys,
    how="left",
    indicator=True
)

matched = merged[merged["_merge"] == "both"].copy()

print("Manual sample rows:", len(manual))
print("Matched reviews:", len(matched))
print("Unmatched reviews:", len(manual) - len(matched))

if matched.empty:
    print("No reviews matched. Check date and review text formats.")
    raise SystemExit

# Evaluate only categories shared by manual and automated tagging
categories = [
    "GPS / tracking / ETA",
    "App stability / update",
    "Pickup / trip reliability",
    "OTP / login / registration",
    "Booking / scheduling",
    "Notifications",
    "UI / usability",
    "Support / communication",
]

# Category-level precision, recall and F1
rows = []

for category in categories:
    tp = sum(
        category in m and category in a
        for m, a in zip(matched["manual_set"], matched["auto_set"])
    )
    fp = sum(
        category not in m and category in a
        for m, a in zip(matched["manual_set"], matched["auto_set"])
    )
    fn = sum(
        category in m and category not in a
        for m, a in zip(matched["manual_set"], matched["auto_set"])
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    rows.append({
        "category": category,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": round(f1 * 100, 2),
    })

metrics = pd.DataFrame(rows)
metrics.to_csv("category_validation_metrics.csv", index=False)

# Save review-level comparison for manual inspection
matched["manual_categories"] = matched["manual_set"].apply(
    lambda x: "; ".join(sorted(x))
)
matched["automated_categories"] = matched["auto_set"].apply(
    lambda x: "; ".join(sorted(x))
)
matched["exact_category_match"] = (
    matched["manual_set"] == matched["auto_set"]
)

matched.to_csv("validation_comparison.csv", index=False)

print("\nCategory validation metrics:")
print(metrics.to_string(index=False))

print(
    "\nExact category match rate:",
    round(matched["exact_category_match"].mean() * 100, 2),
    "%"
)
print("Saved category_validation_metrics.csv")
print("Saved validation_comparison.csv")