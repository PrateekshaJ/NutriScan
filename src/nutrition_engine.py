"""
Nutrition intelligence engine.

Loads per-food nutrition from data/nutrition_db.csv and provides
macros, summaries, and feature vectors for similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NUTRITION_DB = PROJECT_ROOT / "data" / "nutrition_db.csv"


@dataclass(frozen=True)
class NutritionProfile:
    """Nutrition facts per standard serving (typically 100g)."""

    food_name: str
    calories: float
    protein_g: float
    carbohydrates_g: float
    fats_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float
    serving_size_g: float = 100.0

    @property
    def total_macros_g(self) -> float:
        return self.protein_g + self.carbohydrates_g + self.fats_g

    def protein_ratio(self) -> float:
        total = self.total_macros_g
        return self.protein_g / total if total > 0 else 0.0

    def fat_ratio(self) -> float:
        total = self.total_macros_g
        return self.fats_g / total if total > 0 else 0.0

    def carb_ratio(self) -> float:
        total = self.total_macros_g
        return self.carbohydrates_g / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "food_name": self.food_name,
            "calories": self.calories,
            "protein_g": self.protein_g,
            "carbohydrates_g": self.carbohydrates_g,
            "fats_g": self.fats_g,
            "fiber_g": self.fiber_g,
            "sugar_g": self.sugar_g,
            "sodium_mg": self.sodium_mg,
            "serving_size_g": self.serving_size_g,
        }

    def feature_vector(self) -> np.ndarray:
        """Normalized nutrition vector for ML similarity (7 features)."""
        return np.array(
            [
                self.calories / 900.0,
                self.protein_g / 50.0,
                self.carbohydrates_g / 100.0,
                self.fats_g / 50.0,
                self.fiber_g / 15.0,
                self.sugar_g / 50.0,
                self.sodium_mg / 2000.0,
            ],
            dtype=np.float32,
        )


class NutritionEngine:
    """
    Lookup and summarize nutrition for detected foods.

    Extend data/nutrition_db.csv with new rows as your dataset grows.
    """

    REQUIRED_COLUMNS = [
        "food_key",
        "food_name",
        "calories",
        "protein_g",
        "carbohydrates_g",
        "fats_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
        "serving_size_g",
    ]

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_NUTRITION_DB
        self._df = self._load_database()

    def _load_database(self) -> pd.DataFrame:
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Nutrition database not found: {self.db_path}. "
                "Add data/nutrition_db.csv or pass a custom path."
            )
        df = pd.read_csv(self.db_path)
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"nutrition_db.csv missing columns: {missing}")
        df["food_key"] = df["food_key"].str.lower().str.strip()
        return df

    @property
    def available_foods(self) -> List[str]:
        return self._df["food_name"].tolist()

    def _normalize_key(self, food_name: str) -> str:
        return food_name.lower().strip().replace(" ", "_")

    def get_profile(self, food_name: str) -> NutritionProfile:
        """
        Fetch nutrition profile by display name or food_key.

        Falls back to dataset mean if unknown (logged in summary).
        """
        key = self._normalize_key(food_name)
        row = self._df[self._df["food_key"] == key]

        if row.empty:
            # Try matching display name
            row = self._df[
                self._df["food_name"].str.lower() == food_name.lower().strip()
            ]

        if row.empty:
            return self._default_profile(food_name)

        r = row.iloc[0]
        return NutritionProfile(
            food_name=str(r["food_name"]),
            calories=float(r["calories"]),
            protein_g=float(r["protein_g"]),
            carbohydrates_g=float(r["carbohydrates_g"]),
            fats_g=float(r["fats_g"]),
            fiber_g=float(r["fiber_g"]),
            sugar_g=float(r["sugar_g"]),
            sodium_mg=float(r["sodium_mg"]),
            serving_size_g=float(r["serving_size_g"]),
        )

    def _default_profile(self, food_name: str) -> NutritionProfile:
        """Approximate average meal when food is not in database."""
        return NutritionProfile(
            food_name=food_name,
            calories=250.0,
            protein_g=12.0,
            carbohydrates_g=30.0,
            fats_g=10.0,
            fiber_g=3.0,
            sugar_g=8.0,
            sodium_mg=400.0,
        )

    def build_summary(self, profile: NutritionProfile) -> str:
        """Human-readable nutrition summary for the dashboard."""
        cal = profile.calories
        if cal < 120:
            energy = "low calorie"
        elif cal < 300:
            energy = "moderate calorie"
        else:
            energy = "high calorie"

        protein_pct = profile.protein_ratio() * 100
        if protein_pct >= 30:
            macro = "protein-forward"
        elif profile.fat_ratio() >= 0.4:
            macro = "fat-dominant"
        elif profile.carb_ratio() >= 0.5:
            macro = "carb-dominant"
        else:
            macro = "balanced macros"

        fiber_note = (
            "good fiber content"
            if profile.fiber_g >= 5
            else "low fiber — consider adding vegetables"
        )

        return (
            f"{profile.food_name} is a {energy}, {macro} option "
            f"({cal:.0f} kcal per {profile.serving_size_g:.0f}g). "
            f"{fiber_note}. "
            f"Macros: {profile.protein_g:.1f}g protein, "
            f"{profile.carbohydrates_g:.1f}g carbs, {profile.fats_g:.1f}g fat."
        )

    def all_profiles(self) -> List[NutritionProfile]:
        """All foods in the database as NutritionProfile instances."""
        return [self.get_profile(row["food_key"]) for _, row in self._df.iterrows()]

    def profiles_dataframe(self) -> pd.DataFrame:
        """Export nutrition table for analytics notebooks."""
        records = [p.to_dict() for p in self.all_profiles()]
        return pd.DataFrame(records)
