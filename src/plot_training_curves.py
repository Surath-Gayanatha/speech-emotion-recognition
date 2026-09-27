import json
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

HISTORY_PATH = "results/cnn_bilstm_mha_specaug/history.json"
OUTPUT_DIR = "results/cnn_bilstm_mha_specaug"

ACCURACY_OUTPUT = os.path.join(
    OUTPUT_DIR, "accuracy_curve.png"
)

LOSS_OUTPUT = os.path.join(
    OUTPUT_DIR, "loss_curve.png"
)

# --------------------------------------------------
# Load training history
# --------------------------------------------------

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

train_accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

train_loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(train_accuracy) + 1)

# --------------------------------------------------
# Accuracy Curve
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    [x * 100 for x in train_accuracy],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    [x * 100 for x in val_accuracy],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(
    "Training and Validation Accuracy - CNN-BiLSTM-MHA-SpecAugment"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    ACCURACY_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Loss Curve
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Training and Validation Loss - CNN-BiLSTM-MHA-SpecAugment"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    LOSS_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("==============================================")
print("TRAINING CURVES GENERATED")
print("==============================================")
print(f"Accuracy curve : {ACCURACY_OUTPUT}")
print(f"Loss curve     : {LOSS_OUTPUT}")
print(f"Epochs         : {len(train_accuracy)}")
print(f"Final Train Accuracy : {train_accuracy[-1] * 100:.2f}%")
print(f"Final Val Accuracy   : {val_accuracy[-1] * 100:.2f}%")
print("==============================================")