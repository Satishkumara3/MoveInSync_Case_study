
from google_play_scraper import reviews_all, Sort
import pandas as pd

app_id = "com.moveinsync.ets"

print("Fetching MoveInSync reviews...")

reviews = reviews_all(
    app_id,
    sleep_milliseconds=1000,
    lang="en",
    country="in",
    sort=Sort.NEWEST
)

df = pd.DataFrame(reviews)

# Keep useful columns for your case study
columns = [
    "content",
    "score",
    "at",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "replyContent",
    "repliedAt"
]

df = df[[c for c in columns if c in df.columns]]

df = df.rename(columns={
    "content": "review_text",
    "score": "rating",
    "at": "review_date",
    "thumbsUpCount": "helpful_votes",
    "reviewCreatedVersion": "app_version",
    "replyContent": "developer_reply",
    "repliedAt": "reply_date"
})

df.to_csv(
    "moveinsync_reviews.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved {len(df)} reviews to moveinsync_reviews.csv")