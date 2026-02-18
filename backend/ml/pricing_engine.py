"""
Agri-Trust | Pricing Engine
=============================
RandomForestRegressor model that predicts fair market valuation for
bulk agricultural produce.

Inputs:
  1. AI Quality Score    (0.0 – 1.0)  from the Quality Grader
  2. Bulk Volume         (Tons)        shipment size
  3. Regional Market Idx (0.5 – 2.0)  local demand multiplier
  4. Seasonality Factor  (0.5 – 1.5)  harvest-season adjustment

Output:
  Verified Valuation (₹ per ton)
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


class PricingEngine:
    """
    ML-powered pricing model for agricultural commodities.

    Uses a RandomForestRegressor trained on market data to produce
    fair, transparent valuations free from middleman manipulation.
    """

    FEATURE_NAMES = [
        "quality_score",
        "bulk_volume_tons",
        "regional_market_index",
        "seasonality_factor",
    ]

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "models", "pricing_model.joblib"
    )

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self._is_trained = False

        # Auto-train with synthetic data if no saved model exists
        if os.path.exists(self.MODEL_PATH):
            self.load()
        else:
            self._train_with_synthetic_data()

    # ── Synthetic Training Data ───────────────────────────────────────────
    def _train_with_synthetic_data(self):
        """
        Generates realistic synthetic market data and trains the model.

        Pricing formula (ground truth):
            base_price = 25000
            price = base_price × quality^1.5 × market_idx × seasonality
                    × (1 + log2(volume) × 0.02)
            + noise
        """
        np.random.seed(42)
        n_samples = 5000

        quality = np.random.uniform(0.1, 1.0, n_samples)
        volume = np.random.uniform(1, 500, n_samples)
        market_idx = np.random.uniform(0.5, 2.0, n_samples)
        seasonality = np.random.uniform(0.5, 1.5, n_samples)

        X = np.column_stack([quality, volume, market_idx, seasonality])

        # Ground-truth pricing model
        base_price = 25000
        price = (
            base_price
            * np.power(quality, 1.5)
            * market_idx
            * seasonality
            * (1 + np.log2(volume + 1) * 0.02)
        )
        # Add realistic noise (±5%)
        noise = np.random.normal(1.0, 0.05, n_samples)
        y = price * noise

        self.train(X, y)

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the RandomForestRegressor on historical pricing data.

        Args:
            X: (n_samples, 4) array of features
            y: (n_samples,) target prices
        """
        self.model.fit(X, y)
        self._is_trained = True

        # Cross-validation score
        scores = cross_val_score(
            self.model, X, y, cv=5, scoring="r2"
        )
        print(f"📊  Pricing Engine R² = {scores.mean():.4f} (±{scores.std():.4f})")

    # ── Prediction ────────────────────────────────────────────────────────
    def predict_price(
        self,
        quality_score: float,
        bulk_volume_tons: float = 10.0,
        regional_market_index: float = 1.0,
        seasonality_factor: float = 1.0,
    ) -> int:
        """
        Predicts the fair Verified Valuation (₹/ton) for a crop shipment.

        Returns:
            Integer price in ₹ per ton.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        features = np.array([[
            quality_score,
            bulk_volume_tons,
            regional_market_index,
            seasonality_factor,
        ]])

        predicted = self.model.predict(features)[0]
        return max(int(round(predicted)), 0)

    # ── Feature Importance ────────────────────────────────────────────────
    def get_feature_importance(self) -> dict:
        """Returns feature importances from the trained model."""
        if not self._is_trained:
            return {}
        importances = self.model.feature_importances_
        return {
            name: round(float(imp), 4)
            for name, imp in zip(self.FEATURE_NAMES, importances)
        }

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)
        print(f"💾  Model saved to {self.MODEL_PATH}")

    def load(self):
        """Load model from disk."""
        self.model = joblib.load(self.MODEL_PATH)
        self._is_trained = True
        print(f"📂  Model loaded from {self.MODEL_PATH}")


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = PricingEngine()

    test_cases = [
        {"quality_score": 0.95, "bulk_volume_tons": 50,  "regional_market_index": 1.2, "seasonality_factor": 1.1},
        {"quality_score": 0.75, "bulk_volume_tons": 100, "regional_market_index": 1.0, "seasonality_factor": 0.9},
        {"quality_score": 0.40, "bulk_volume_tons": 20,  "regional_market_index": 0.8, "seasonality_factor": 0.7},
    ]

    print("\n🧪  Price Predictions:")
    for tc in test_cases:
        price = engine.predict_price(**tc)
        print(f"  Quality={tc['quality_score']:.2f}  Vol={tc['bulk_volume_tons']}T  → ₹{price:,}/ton")

    print("\n📈  Feature Importance:")
    for feat, imp in engine.get_feature_importance().items():
        print(f"  {feat}: {imp}")
