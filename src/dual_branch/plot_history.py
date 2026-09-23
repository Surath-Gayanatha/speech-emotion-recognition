import json
import os
import matplotlib.pyplot as plt

HISTORY_PATH = "results/dual_branch/history.json"
OUTPUT_DIR = "results/dual_branch"

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = range(1, len(history["loss"]) + 1)

# ============================================================
# ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dual-Branch Model - Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Dual-Branch Model - Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# BEST EPOCH
# ============================================================

best_epoch = history["val_loss"].index(
    min(history["val_loss"])
) + 1

print("=" * 65)
print("VALIDATION ANALYSIS")
print("=" * 65)

print(f"Total epochs trained : {len(history['loss'])}")
print(f"Best epoch           : {best_epoch}")

print(
    f"Best validation loss : "
    f"{history['val_loss'][best_epoch - 1]:.4f}"
)

print(
    f"Validation accuracy  : "
    f"{history['val_accuracy'][best_epoch - 1] * 100:.2f}%"
)

print(
    f"Training accuracy    : "
    f"{history['accuracy'][best_epoch - 1] * 100:.2f}%"
)

gap = (
    history["accuracy"][best_epoch - 1]
    -
    history["val_accuracy"][best_epoch - 1]
) * 100

print(
    f"Train-Val gap        : "
    f"{gap:.2f} percentage points"
)

print("\nSaved:")
print("results/dual_branch/accuracy_curve.png")
print("results/dual_branch/loss_curve.png")