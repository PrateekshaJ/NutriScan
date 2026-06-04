"""
Health score system (0–100).

Scores foods using calories, protein ratio, fat level, and macro balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.nutrition_engine import NutritionProfile


@dataclass(frozen=True)
class HealthScoreResult:
    """Composite health score with interpretable sub-scores."""

    overall_score: float
    calorie_score: float
    protein_score: float
    fat_score: float
    balance_score: float
    grade: str
    insights: List[str]

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall_score": self.overall_score,
            "calorie_score": self.calorie_score,
            "protein_score": self.protein_score,
            "fat_score": self.fat_score,
            "balance_score": self.balance_score,
            "grade": self.grade,
        }


class HealthScoreEngine:
    """
    Compute a 0–100 health score from a NutritionProfile.

    Weights (sum to 1.0):
        - calories: 25%
        - protein ratio: 30%
        - fat level: 20%
        - nutritional balance: 25%
    """

    WEIGHTS = {
        "calories": 0.25,
        "protein": 0.30,
        "fat": 0.20,
        "balance": 0.25,
    }

    def score(self, profile: NutritionProfile) -> HealthScoreResult:
        cal_s = self._score_calories(profile.calories)
        prot_s = self._score_protein_ratio(profile)
        fat_s = self._score_fat_level(profile)
        bal_s = self._score_balance(profile)

        overall = (
            cal_s * self.WEIGHTS["calories"]
            + prot_s * self.WEIGHTS["protein"]
            + fat_s * self.WEIGHTS["fat"]
            + bal_s * self.WEIGHTS["balance"]
        )
        overall = round(min(100.0, max(0.0, overall)), 1)

        insights = self._build_insights(profile, cal_s, prot_s, fat_s, bal_s)
        return HealthScoreResult(
            overall_score=overall,
            calorie_score=round(cal_s, 1),
            protein_score=round(prot_s, 1),
            fat_score=round(fat_s, 1),
            balance_score=round(bal_s, 1),
            grade=self._grade(overall),
            insights=insights,
        )

    def _score_calories(self, calories: float) -> float:
        """Peak score around 150–350 kcal per serving; penalize extremes."""
        if calories <= 80:
            return 70.0
        if calories <= 150:
            return 85.0 + (calories - 80) / 70 * 10
        if calories <= 350:
            return 95.0 - abs(calories - 250) / 200 * 15
        if calories <= 550:
            return max(40.0, 80.0 - (calories - 350) / 4)
        return max(15.0, 50.0 - (calories - 550) / 10)

    def _score_protein_ratio(self, profile: NutritionProfile) -> float:
        ratio = profile.protein_ratio()
        # Ideal band ~25–40% of macro calories from protein context
        if 0.22 <= ratio <= 0.45:
            return 90.0 + (1 - abs(ratio - 0.32) / 0.2) * 10
        if ratio < 0.15:
            return 40.0 + ratio / 0.15 * 30
        if ratio > 0.55:
            return 75.0
        return 65.0 + ratio * 40

    def _score_fat_level(self, profile: NutritionProfile) -> float:
        fat_ratio = profile.fat_ratio()
        fat_g = profile.fats_g
        if fat_g <= 8 and fat_ratio <= 0.35:
            return 92.0
        if fat_g <= 15 and fat_ratio <= 0.4:
            return 80.0
        if fat_g <= 25:
            return 65.0
        return max(25.0, 60.0 - (fat_g - 25) * 2)

    def _score_balance(self, profile: NutritionProfile) -> float:
        """Reward fiber, penalize sugar and sodium."""
        score = 70.0
        if profile.fiber_g >= 6:
            score += 15
        elif profile.fiber_g >= 3:
            score += 8
        if profile.sugar_g > 20:
            score -= min(25, (profile.sugar_g - 20) * 1.2)
        elif profile.sugar_g < 8:
            score += 5
        if profile.sodium_mg > 800:
            score -= min(20, (profile.sodium_mg - 800) / 50)
        # Macro spread — avoid one macro dominating excessively
        ratios = [profile.protein_ratio(), profile.carb_ratio(), profile.fat_ratio()]
        dominance = max(ratios)
        if dominance > 0.65:
            score -= 12
        return min(100.0, max(0.0, score))

    def _grade(self, score: float) -> str:
        if score >= 85:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 55:
            return "Fair"
        if score >= 40:
            return "Needs Improvement"
        return "Poor"

    def _build_insights(
        self,
        profile: NutritionProfile,
        cal_s: float,
        prot_s: float,
        fat_s: float,
        bal_s: float,
    ) -> List[str]:
        insights: List[str] = []
        if cal_s < 60:
            insights.append("Calorie density is high — watch portion size.")
        if prot_s < 60:
            insights.append("Protein content is low relative to total macros.")
        if fat_s < 60:
            insights.append("Fat content is elevated — prefer leaner preparation.")
        if bal_s < 60:
            insights.append("Fiber is low or sugar/sodium is high — balance with whole foods.")
        if profile.fiber_g >= 5:
            insights.append("Good fiber supports digestion and satiety.")
        if not insights:
            insights.append("Well-rounded nutritional profile for everyday meals.")
        return insights

    def sub_scores_for_chart(self, result: HealthScoreResult) -> Dict[str, float]:
        """Labels and values for radar/bar charts in Streamlit."""
        return {
            "Calories": result.calorie_score,
            "Protein": result.protein_score,
            "Fat": result.fat_score,
            "Balance": result.balance_score,
        }
