# ============================================================================
#  FarmPulse — Plant Disease Detection Model Training (Local optimized v2)
# ============================================================================
#
#  INSTRUCTIONS:
#  1. Ensure you have requirements installed:
#     pip install tensorflow tensorflowjs opendatasets numpy
#  2. Run the script: python ml_training/train_disease_model.py
#  3. You will need a Kaggle account and API key (kaggle.json) for the download.
#
# ============================================================================

import os
import json
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflowjs as tfjs
import opendatasets as od

# ── Step 1: Configuration & Paths ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"📂 Project Root: {BASE_DIR}")
print(f"💾 Model Save Path: {SAVE_DIR}")

# ── Step 2: Download Dataset ────────────────────────────────────────────────
# Using Kaggle via opendatasets
print("\n📥 Checking for PlantVillage Dataset...")
dataset_url = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
DATASET_ROOT = os.path.join(DATA_DIR, "plantvillage-dataset")
DATASET_DIR = os.path.join(DATASET_ROOT, "plantvillage dataset", "color")

if not os.path.exists(DATASET_DIR):
    print("   Dataset not found. Starting download (requires kaggle.json)...")
    od.download(dataset_url, data_dir=DATA_DIR)
else:
    print("   ✅ Dataset already exists.")

# Verify path again after download (opendatasets might create extra folders)
if not os.path.exists(DATASET_DIR):
    # Try alternative path searching
    found = False
    for root, dirs, files in os.walk(DATA_DIR):
        for d in dirs:
            if "color" in d.lower() or "Color" in d:
                DATASET_DIR = os.path.join(root, d)
                found = True
                break
        if found: break

if not os.path.exists(DATASET_DIR):
    print(f"❌ Error: Could not find 'color' directory in {DATA_DIR}")
    exit(1)

classes = sorted(os.listdir(DATASET_DIR))
print(f"\n✅ Found {len(classes)} disease classes")

# ── Step 3: Efficient Data Preparation (tf.data) ──────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32

print("\n📊 Loading and optimizing data pipeline...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Extract class names
class_names = train_ds.class_names

# Optimization for performance
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    # Rescale to [0,1]
    return image / 255.0, label

# Training data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# Cache and Prefetch for maximum speed
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

NUM_CLASSES = len(class_names)
print(f"✅ Data pipeline ready: {NUM_CLASSES} classes")

# ── Step 4: Build Model ─────────────────────────────────────────────────────
print("\n🏗️ Building MobileNetV2...")
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(f"✅ Model ready: {model.count_params():,} params")

# ── Step 5: Train ────────────────────────────────────────────────────────────
cbs = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

print("\n🚀 Phase 1: Training (frozen base)...")
model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=cbs)

print("\n🔧 Phase 2: Fine-tuning...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=cbs)

# ── Step 6: Save Model and Metadata ─────────────────────────────────────────
print("\n💾 Saving results...")

# Generate and save class labels metadata
class_labels = {}
for idx, name in enumerate(class_names):
    parts = name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
    class_labels[str(idx)] = {
        "class_name": name,
        "crop": crop,
        "disease": disease,
        "display_name": f"{crop} — {disease}",
        "is_healthy": "healthy" in disease.lower()
    }

# Save metadata JSONs
with open(os.path.join(SAVE_DIR, "class_labels.json"), "w") as f:
    json.dump(class_labels, f, indent=2)

disease_info = {}
for idx_str, label in class_labels.items():
    key = label["class_name"]
    if label["is_healthy"]:
        disease_info[key] = {
            "severity": "low",
            "symptoms": "No disease symptoms detected. Plant appears healthy.",
            "cause": "N/A — Your crop is in good health!",
            "solution": "Continue current management practices.",
            "prevention": "Regular scouting every 7 days.",
            "yieldLoss": "0%", "treatmentCost": "₹0"
        }
    else:
        disease_info[key] = {
            "severity": "high",
            "symptoms": f"Visual signs of {label['disease']} detected on {label['crop']}.",
            "cause": f"Pathogen causing {label['disease']} in {label['crop']}.",
            "solution": f"Consult agricultural extension for {label['disease']} treatment.",
            "prevention": "Use resistant varieties. Crop rotation. Field hygiene.",
            "yieldLoss": "20-50%", "treatmentCost": "₹800-1500/acre"
        }

with open(os.path.join(SAVE_DIR, "disease_info.json"), "w") as f:
    json.dump(disease_info, f, indent=2)

# Convert to TensorFlow.js for the frontend
print("📦 Converting to TensorFlow.js...")
tfjs_dir = os.path.join(SAVE_DIR, "tfjs_temp")
os.makedirs(tfjs_dir, exist_ok=True)
tfjs.converters.save_keras_model(model, tfjs_dir)

# Move tfjs files
for fname in os.listdir(tfjs_dir):
    shutil.move(os.path.join(tfjs_dir, fname), os.path.join(SAVE_DIR, fname))
shutil.rmtree(tfjs_dir)

print(f"\n✅ SUCCESS! All files saved to: {SAVE_DIR}")

