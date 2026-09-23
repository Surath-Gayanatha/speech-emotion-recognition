import os
import json
import numpy as np

MFCC_DIR = "data/processed"
LOGMEL_DIR = "data/processed/logmel"
OUT_DIR = "data/processed/dual_branch"

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("DUAL-BRANCH DATA PREPARATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load MFCC data
# ---------------------------------------------------------
print("\n[1] Loading MFCC data...")

mfcc = np.load(
    os.path.join(MFCC_DIR, "features.npy")
)

mfcc_labels = np.load(
    os.path.join(MFCC_DIR, "labels.npy")
)

mfcc_filenames = np.load(
    os.path.join(MFCC_DIR, "filenames.npy"),
    allow_pickle=True
)

print(f"MFCC features : {mfcc.shape}")
print(f"MFCC labels   : {mfcc_labels.shape}")
print(f"MFCC filenames: {mfcc_filenames.shape}")


# ---------------------------------------------------------
# 2. Load Log-Mel split data
# ---------------------------------------------------------
print("\n[2] Loading Log-Mel data...")

splits = {}

for split in ["train", "validation", "test"]:

    feature_file = os.path.join(
        LOGMEL_DIR,
        f"{split}_features.npy"
    )

    label_file = os.path.join(
        LOGMEL_DIR,
        f"{split}_labels.npy"
    )

    filename_file = os.path.join(
        LOGMEL_DIR,
        f"{split}_filenames.npy"
    )

    splits[split] = {
        "features": np.load(feature_file),
        "labels": np.load(label_file),
        "filenames": np.load(
            filename_file,
            allow_pickle=True
        )
    }

    print(
        f"{split:12s}: "
        f"features={splits[split]['features'].shape}, "
        f"labels={splits[split]['labels'].shape}, "
        f"filenames={splits[split]['filenames'].shape}"
    )


# ---------------------------------------------------------
# 3. Create MFCC filename -> index mapping
# ---------------------------------------------------------
print("\n[3] Creating MFCC filename index...")

mfcc_index = {
    str(filename): i
    for i, filename in enumerate(mfcc_filenames)
}

print(f"MFCC index entries: {len(mfcc_index)}")


# ---------------------------------------------------------
# 4. Verify duplicates
# ---------------------------------------------------------
print("\n[4] Checking duplicate filenames...")

unique_mfcc = len(set(map(str, mfcc_filenames)))

if unique_mfcc != len(mfcc_filenames):
    raise ValueError(
        f"Duplicate MFCC filenames detected! "
        f"{len(mfcc_filenames) - unique_mfcc} duplicates."
    )

print("✓ No duplicate MFCC filenames")


# ---------------------------------------------------------
# 5. Align MFCC according to Log-Mel filenames
# ---------------------------------------------------------
print("\n[5] Aligning MFCC with Log-Mel splits...")

aligned = {}

for split in ["train", "validation", "test"]:

    logmel_data = splits[split]

    filenames = logmel_data["filenames"]

    indices = []

    missing = []

    for filename in filenames:

        filename = str(filename)

        if filename not in mfcc_index:
            missing.append(filename)
        else:
            indices.append(mfcc_index[filename])

    if missing:
        print("\nERROR: Missing MFCC files:")
        for filename in missing[:20]:
            print("  ", filename)

        raise ValueError(
            f"{len(missing)} Log-Mel filenames "
            f"were not found in MFCC data."
        )

    indices = np.array(indices)

    aligned_mfcc = mfcc[indices]

    aligned_mfcc_labels = mfcc_labels[indices]

    # -----------------------------------------------------
    # Label consistency check
    # -----------------------------------------------------
    if not np.array_equal(
        aligned_mfcc_labels,
        logmel_data["labels"]
    ):
        mismatch = np.sum(
            aligned_mfcc_labels != logmel_data["labels"]
        )

        raise ValueError(
            f"{split}: {mismatch} label mismatches detected!"
        )

    # -----------------------------------------------------
    # Filename order verification
    # -----------------------------------------------------
    aligned_names = mfcc_filenames[indices]

    if not np.array_equal(
        aligned_names.astype(str),
        filenames.astype(str)
    ):
        raise ValueError(
            f"{split}: Filename alignment verification failed!"
        )

    aligned[split] = {
        "mfcc": aligned_mfcc,
        "logmel": logmel_data["features"],
        "labels": logmel_data["labels"],
        "filenames": filenames
    }

    print(
        f"✓ {split:12s} "
        f"MFCC={aligned_mfcc.shape} "
        f"LogMel={logmel_data['features'].shape} "
        f"Labels={logmel_data['labels'].shape}"
    )


# ---------------------------------------------------------
# 6. Save aligned datasets
# ---------------------------------------------------------
print("\n[6] Saving aligned datasets...")

for split in ["train", "validation", "test"]:

    data = aligned[split]

    np.save(
        os.path.join(
            OUT_DIR,
            f"{split}_mfcc.npy"
        ),
        data["mfcc"]
    )

    np.save(
        os.path.join(
            OUT_DIR,
            f"{split}_logmel.npy"
        ),
        data["logmel"]
    )

    np.save(
        os.path.join(
            OUT_DIR,
            f"{split}_labels.npy"
        ),
        data["labels"]
    )

    np.save(
        os.path.join(
            OUT_DIR,
            f"{split}_filenames.npy"
        ),
        data["filenames"]
    )


# ---------------------------------------------------------
# 7. Final verification
# ---------------------------------------------------------
print("\n[7] Final verification")

summary = {}

for split in ["train", "validation", "test"]:

    data = aligned[split]

    print(f"\n{split.upper()}")

    print(f"  MFCC    : {data['mfcc'].shape}")
    print(f"  Log-Mel : {data['logmel'].shape}")
    print(f"  Labels  : {data['labels'].shape}")

    assert data["mfcc"].shape[0] == data["logmel"].shape[0]
    assert data["mfcc"].shape[0] == data["labels"].shape[0]

    summary[split] = {
        "mfcc_shape": list(data["mfcc"].shape),
        "logmel_shape": list(data["logmel"].shape),
        "labels_shape": list(data["labels"].shape)
    }

# ---------------------------------------------------------
# 8. Save metadata
# ---------------------------------------------------------
metadata = {
    "dataset": "CREMA-D",
    "alignment_method": "MFCC filenames matched to Log-Mel split filenames",
    "actor_independent_split": True,
    "train_only_normalization": True,
    "splits": summary
}

with open(
    os.path.join(OUT_DIR, "metadata.json"),
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput directory:")
print(OUT_DIR)

print("\nFiles created:")

for filename in sorted(os.listdir(OUT_DIR)):
    print("  ", filename)

print("\n✓ Filename alignment passed")
print("✓ Label consistency passed")
print("✓ Shape consistency passed")
print("✓ Actor-independent split preserved")