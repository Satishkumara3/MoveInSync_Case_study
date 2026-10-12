
import pandas as pd

# Load and clean the dataset
df = pd.read_csv("moveinsync_reviews.csv")

df["review_text"] = df["review_text"].fillna("").astype(str)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["review_date"] = pd.to_datetime(
    df["review_date"], errors="coerce"
)

df = df.dropna(subset=["rating", "review_date"])
df = df[df["rating"].between(1, 5)].copy()

df["review_text"] = df["review_text"].str.strip()
df["text_lower"] = df["review_text"].str.lower()
df["negative"] = df["rating"] <= 2


# Support / communication keywords
support_keywords = [
    "support",
    "customer care",
    "customer service",
    "helpdesk",
    "call option",
    "calling option",
    "call facility",
    "chat option",
    "chat facility",
    "contact driver",
    "contact support",
    "no call",
    "no chat",

    # Retained from the original rules
    "help desk",
    "complaint",
    "no response",
]


# Category keyword rules
categories = {
    "App stability / update": [
        "crash", "crashing", "reinstall", "update",
        "install", "freeze", "freezing", "hang",
        "not opening", "unable to open", "app won't open",
        "app not opening"
    ],

    "Booking": [
        "booking", "book a cab", "book a ride",
        "cancel booking", "wrong date", "reservation"
    ],

    "GPS / tracking / ETA": [
        "gps", "location", "tracking", "track cab",
        "track the cab", "eta", "live location"
    ],

    "Pickup / trip reliability": [
        "pickup",
    "pick up",
    "cab late",
    "cab delayed",
    "cab delay",
    "driver late",
    "vehicle late",
    "missed pickup",
    "missed the cab",
    "no show",
    "no-show",
    "trip delayed",
    "trip delay",
    "transport delayed",
    "wrong routing",
    "bad routing",
    "very bad routing",
    "managed routing",
    "routing issue",
    "routing problem",
    "route issue",
    "route problem",
    "drop location",
    "drop locations",
    "cab service",
    "waiting for cab",
    "wait for cab",
    "cab arrived",
    ],

    "OTP / login / verification": [
        "otp", "login", "log in", "logged out",
        "verification", "password", "unable to login"
    ],

    "Notifications": [
        "notification", "notifications", "alert",
        "reminder", "not notified", "notification late"
    ],

    "UI / usability": [
        "feature", "click", "button", "interface",
        "annoying", "difficult", "confusing",
        "user friendly", "too many steps"
    ],

    # Uses the updated Support keyword list
    "Support": support_keywords
}


# Categorization function
def get_categories(text):
    text = str(text).lower()

    return [
        category
        for category, keywords in categories.items()
        if any(keyword in text for keyword in keywords)
    ]


# Apply categories to every review
df["categories"] = df["text_lower"].apply(get_categories)

# Export tagged reviews
df.to_csv("tagged_reviews.csv", index=False)


# Generate category summary
summary = []

for category in categories:
    mask = df["categories"].apply(
        lambda tags: category in tags
    )

    subset = df[mask]

    if subset.empty:
        continue

    summary.append({
        "category": category,
        "tagged_reviews": len(subset),
        "negative_reviews": int(subset["negative"].sum()),
        "negative_share_pct": round(
            subset["negative"].mean() * 100, 2
        ),
        "average_rating": round(
            subset["rating"].mean(), 2
        )
    })

summary_df = pd.DataFrame(summary)

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        "tagged_reviews", ascending=False
    )

print("\n=== CATEGORY SUMMARY ===")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    "validated_category_summary.csv", index=False
)


# Analyze the latest 12 months of reviews
if not df.empty:
    latest_date = df["review_date"].max()
    cutoff = latest_date - pd.DateOffset(months=12)

    recent = df[df["review_date"] >= cutoff].copy()

    print("\n=== DATASET TIME WINDOW ===")
    print("Latest review:", latest_date)
    print("12-month cutoff:", cutoff)
    print("Recent review count:", len(recent))

    print(
        "Recent average rating:",
        round(recent["rating"].mean(), 2)
    )

    print(
        "Recent 1-2 star share:",
        round(recent["negative"].mean() * 100, 2),
        "%"
    )

    recent_rows = []

    for category in categories:
        mask = recent["categories"].apply(
            lambda tags: category in tags
        )

        subset = recent[mask]

        if subset.empty:
            continue

        recent_rows.append({
            "category": category,
            "recent_reviews": len(subset),
            "negative_reviews": int(
                subset["negative"].sum()
            ),
            "negative_share_pct": round(
                subset["negative"].mean() * 100, 2
            ),
            "average_rating": round(
                subset["rating"].mean(), 2
            )
        })

    recent_summary = pd.DataFrame(recent_rows)

    if not recent_summary.empty:
        recent_summary = recent_summary.sort_values(
            "recent_reviews", ascending=False
        )

        print("\n=== RECENT 12-MONTH CATEGORIES ===")
        print(recent_summary.to_string(index=False))

        recent_summary.to_csv(
            "recent_category_summary.csv", index=False
        )


# Print examples of low-rated reviews for manual validation
print("\n=== EXAMPLE LOW-RATED REVIEWS ===")

for category in categories:
    mask = (
        df["categories"].apply(
            lambda tags: category in tags
        )
        & df["negative"]
        & (df["review_text"].str.len() > 0)
    )

    examples = df.loc[mask].sort_values(
        "review_date", ascending=False
    ).head(3)

    if not examples.empty:
        print(f"\n--- {category} ---")

        for _, row in examples.iterrows():
            print(
                f"Rating: {row['rating']} | "
                f"Date: {row['review_date'].date()}"
            )
            print(row["review_text"][:400])
            print()


print("\nSaved tagged_reviews.csv")
print("Saved validated_category_summary.csv")
print("Saved recent_category_summary.csv if recent matches exist.")
