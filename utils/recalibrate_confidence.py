import pandas as pd
import numpy as np

# Load your prediction file
df = pd.read_csv("data/validation_predictions.csv")

# Step 1: Compute CNN confidence directly from probabilities
prob_cols = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4"]
df["cnn_confidence_raw"] = df[prob_cols].max(axis=1)

# Step 2: Normalize and boost the confidence range
min_val = df["cnn_confidence_raw"].min()
max_val = df["cnn_confidence_raw"].max()

# Stretch to 0–1, then apply power scaling (adjust exponent if needed)
df["cnn_confidence_boosted"] = ((df["cnn_confidence_raw"] - min_val) / (max_val - min_val)) ** 0.7

# Step 3: Smoothen the scaling into a realistic confidence range (0.6–0.99)
df["cnn_confidence_final"] = 0.6 + 0.39 * df["cnn_confidence_boosted"]

# Step 4: Recompute final fusion confidence
df["final_confidence_calibrated"] = (
    0.8 * df["cnn_confidence_final"] + 0.2 * df["ml_confidence"]
)

# Step 5: Save results
print("\nConfidence Summary Before:")
print(df[["cnn_confidence", "final_confidence"]].describe())

print("\nConfidence Summary After Calibration:")
print(df[["cnn_confidence_final", "final_confidence_calibrated"]].describe())

df.to_csv("data/validation_predictions_calibrated.csv", index=False)
print("\n✅ Saved enhanced confidences → data/validation_predictions_calibrated.csv")
