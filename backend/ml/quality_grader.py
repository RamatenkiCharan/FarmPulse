"""
Agri-Trust | Quality Grading Module
====================================
Uses MobileNetV2 (Transfer Learning) to classify agricultural produce into:
  - 'Export-Grade'   : 90%+ purity
  - 'Local-Market'   : 70-90% purity
  - 'Processing'     : <70% purity

The model loads ImageNet pre-trained weights and adds a custom 3-class
classification head fine-tuned for crop quality assessment.
"""

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io


# ── Grade Definitions ────────────────────────────────────────────────────────
GRADE_LABELS = ["Export-Grade", "Local-Market", "Processing"]

GRADE_THRESHOLDS = {
    "Export-Grade": 0.90,   # 90%+ purity
    "Local-Market": 0.70,   # 70-90% purity
    "Processing":   0.0,    # <70% purity
}

IMAGE_SIZE = (224, 224)


class CropQualityGrader:
    """
    CNN-based crop quality classification using MobileNetV2 backbone.

    In production, call `train()` with labelled crop images, then `save()`
    the weights. For inference, `load()` weights and call `grade()`.
    """

    def __init__(self):
        self.model = self._build_model()

    # ── Model Architecture ────────────────────────────────────────────────
    def _build_model(self) -> Model:
        """
        Constructs MobileNetV2 + custom classification head.

        Architecture:
            MobileNetV2 (frozen) → GlobalAvgPool → Dense(256) → Dropout →
            Dense(128) → Dropout → Dense(3, softmax)
        """
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3),
        )

        # Freeze the base model for transfer learning
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu", name="fc_256")(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu", name="fc_128")(x)
        x = Dropout(0.2)(x)
        predictions = Dense(
            len(GRADE_LABELS), activation="softmax", name="grade_output"
        )(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Preprocessing ─────────────────────────────────────────────────────
    @staticmethod
    def preprocess_image(image_bytes: bytes) -> np.ndarray:
        """
        Converts raw image bytes → preprocessed (1, 224, 224, 3) tensor
        suitable for MobileNetV2 inference.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    # ── Inference ─────────────────────────────────────────────────────────
    def grade(self, image_bytes: bytes) -> dict:
        """
        Runs the full grading pipeline on raw image bytes.

        Returns:
            {
                "quality_grade": "Export-Grade" | "Local-Market" | "Processing",
                "purity_score": 0.0 – 1.0,
                "confidence_scores": {grade: float, ...}
            }
        """
        processed = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed, verbose=0)
        scores = predictions[0]

        # Map softmax probabilities to grades
        confidence_scores = {
            label: round(float(score), 4)
            for label, score in zip(GRADE_LABELS, scores)
        }

        # Determine grade via weighted purity score
        # Purity is the weighted sum biased towards export quality
        purity_score = float(
            scores[0] * 1.0 + scores[1] * 0.8 + scores[2] * 0.4
        )
        purity_score = round(min(purity_score, 1.0), 4)

        if purity_score >= GRADE_THRESHOLDS["Export-Grade"]:
            quality_grade = "Export-Grade"
        elif purity_score >= GRADE_THRESHOLDS["Local-Market"]:
            quality_grade = "Local-Market"
        else:
            quality_grade = "Processing"

        return {
            "quality_grade": quality_grade,
            "purity_score": purity_score,
            "confidence_scores": confidence_scores,
        }

    # ── Persistence ───────────────────────────────────────────────────────
    def save_weights(self, path: str = "models/crop_grader_weights.h5"):
        """Save trained model weights to disk."""
        self.model.save_weights(path)

    def load_weights(self, path: str = "models/crop_grader_weights.h5"):
        """Load pre-trained model weights from disk."""
        self.model.load_weights(path)


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    grader = CropQualityGrader()
    print("✅  CropQualityGrader model built successfully.")
    grader.model.summary()
