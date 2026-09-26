import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs(r'dataset\clean_splits', exist_ok=True)

df_train = pd.read_csv(r'dataset/raw_train.csv')
df_test = pd.read_csv(r'dataset/raw_test.csv')

df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all['clean_text'] = df_all['text'].astype(str).str.strip()

print(f"Raw total samples: {len(df_all)}")
df_dedup = df_all.drop_duplicates(subset=['clean_text']).copy().reset_index(drop=True)
print(f"Deduplicated unique samples: {len(df_dedup)}")

# Stratified split: 70% Train, 15% Val, 15% Test based on label
# Note: stratify on label
train_val, test_df = train_test_split(
    df_dedup, test_size=0.15, random_state=42, stratify=df_dedup['label']
)

# Out of remaining 85%, 15/85 ≈ 0.17647 for Val, 70/85 for Train
train_df, val_df = train_test_split(
    train_val, test_size=0.15 / 0.85, random_state=42, stratify=train_val['label']
)

print(f"Split sizes: Train={len(train_df)} (70.0%), Val={len(val_df)} (15.0%), Test={len(test_df)} (15.0%)")

train_df.to_csv(r'dataset\clean_splits\clean_train.csv', index=False, encoding='utf-8')
val_df.to_csv(r'dataset\clean_splits\clean_val.csv', index=False, encoding='utf-8')
test_df.to_csv(r'dataset\clean_splits\clean_test.csv', index=False, encoding='utf-8')

split_summary = {
    "total_raw": len(df_all),
    "total_unique": len(df_dedup),
    "splits": {
        "train": {
            "count": len(train_df),
            "ratio": 0.70,
            "label_dist": train_df['label'].value_counts().to_dict()
        },
        "val": {
            "count": len(val_df),
            "ratio": 0.15,
            "label_dist": val_df['label'].value_counts().to_dict(),
            "purpose": "Confidence threshold tuning & calibration ONLY"
        },
        "test": {
            "count": len(test_df),
            "ratio": 0.15,
            "label_dist": test_df['label'].value_counts().to_dict(),
            "purpose": "Final held-out evaluation"
        }
    }
}

with open(r'dataset\clean_splits\split_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(split_summary, f, indent=2, ensure_ascii=False)

print("Clean 3-way split created successfully!")

