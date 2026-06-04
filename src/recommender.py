"""
Goal-based AI suggestions and similar-food recommendation engine.

Uses rule-based goal logic plus scikit-learn NearestNeighbors on
nutrition feature vectors for healthier alternative search.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.health_score import HealthScoreEngine, HealthScoreResult
from src.nutrition_engine import NutritionEngine, NutritionProfile


class GoalType(str, Enum):
    WEIGHT_LOSS = "Weight Loss"
    MUSCLE_GAIN = "Muscle Gain"
    HEALTHY_MAINTENANCE = "Healthy Maintenance"


@dataclass(frozen=True)
class GoalAssessment:
    fits_goal: bool
    verdict: str
    improvements: List[str]
    healthier_swaps: List[str]


@dataclass(frozen=True)
class SimilarFood:
    food_name: str
    similarity: float
    calories: float
    health_score: float
    reason: str


@dataclass(frozen=True)
class RecommendationBundle:
    goal: GoalType
    assessment: GoalAssessment
    similar_foods: List[SimilarFood]


class FoodRecommender:
    """
    Goal-based recommendations and ML similarity search.

    Similarity: cosine distance on normalized nutrition vectors;
    prefers neighbors with higher health scores when breaking ties.
    """

    def __init__(
        self,
        nutrition_engine: Optional[NutritionEngine] = None,
        health_engine: Optional[HealthScoreEngine] = None,
        n_neighbors: int = 5,
    ) -> None:
        self.nutrition = nutrition_engine or NutritionEngine()
        self.health = health_engine or HealthScoreEngine()
        self.n_neighbors = n_neighbors
        self._profiles: List[NutritionProfile] = []
        self._keys: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self._health_by_key: Dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        self._profiles = self.nutrition.all_profiles()
        self._keys = [
            self.nutrition._normalize_key(p.food_name) for p in self._profiles
        ]
        self._vectors = np.vstack([p.feature_vector() for p in self._profiles])
        self._nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 3, len(self._profiles)),
            metric="cosine",
            algorithm="brute",
        )
        self._nn.fit(self._vectors)
        for p, key in zip(self._profiles, self._keys):
            self._health_by_key[key] = self.health.score(p).overall_score

    def recommend(
        self,
        profile: NutritionProfile,
        goal: GoalType,
        health_result: Optional[HealthScoreResult] = None,
    ) -> RecommendationBundle:
        health_result = health_result or self.health.score(profile)
        assessment = self._assess_goal(profile, goal, health_result)
        similar = self.find_similar_healthier(profile, exclude_name=profile.food_name)
        return RecommendationBundle(
            goal=goal,
            assessment=assessment,
            similar_foods=similar,
        )

    def _assess_goal(
        self,
        profile: NutritionProfile,
        goal: GoalType,
        health: HealthScoreResult,
    ) -> GoalAssessment:
        improvements: List[str] = []
        swaps: List[str] = []

        if goal == GoalType.WEIGHT_LOSS:
            fits = profile.calories <= 350 and profile.fats_g <= 18
            if profile.calories > 400:
                improvements.append("Reduce portion size or choose a lighter base.")
            if profile.fats_g > 20:
                improvements.append("Lower added fats — grill, bake, or steam instead of fry.")
            if profile.sugar_g > 15:
                improvements.append("Cut sugary sauces and drinks with this meal.")
            swaps.extend(["salad", "grilled chicken", "soup", "yogurt"])

        elif goal == GoalType.MUSCLE_GAIN:
            fits = profile.protein_g >= 20 and profile.calories >= 200
            if profile.protein_g < 18:
                improvements.append("Add lean protein: chicken, fish, legumes, or Greek yogurt.")
            if profile.calories < 180:
                improvements.append("Increase calories with complex carbs and healthy fats.")
            swaps.extend(["grilled chicken", "steak", "rice bowl", "sushi"])

        else:  # HEALTHY_MAINTENANCE
            fits = health.overall_score >= 60
            if health.balance_score < 65:
                improvements.append("Add vegetables or whole grains for fiber and balance.")
            if profile.sodium_mg > 600:
                improvements.append("Reduce sodium — rinse canned goods, limit processed sides.")
            swaps.extend(["salad", "apple", "banana", "soup", "yogurt"])

        if not improvements:
            improvements.append("Maintain current choices; monitor portions and variety.")

        verdict = (
            f"This food {'aligns well' if fits else 'partially fits'} with "
            f"{goal.value}. Health score: {health.overall_score}/100 ({health.grade})."
        )

        return GoalAssessment(
            fits_goal=fits,
            verdict=verdict,
            improvements=improvements[:4],
            healthier_swaps=[s.replace("_", " ").title() for s in swaps[:3]],
        )

    def find_similar_healthier(
        self,
        profile: NutritionProfile,
        exclude_name: Optional[str] = None,
        top_k: int = 4,
    ) -> List[SimilarFood]:
        """
        ML similarity search: nearest foods in nutrition space,
        ranked by similarity then filtered/sorted by health score.
        """
        if self._nn is None or self._vectors is None:
            return []

        query = profile.feature_vector().reshape(1, -1)
        distances, indices = self._nn.kneighbors(query)

        candidates: List[SimilarFood] = []
        source_key = self.nutrition._normalize_key(profile.food_name)
        source_health = self.health.score(profile).overall_score

        for dist, idx in zip(distances[0], indices[0]):
            candidate = self._profiles[idx]
            key = self._keys[idx]
            if key == source_key:
                continue
            if exclude_name and candidate.food_name.lower() == exclude_name.lower():
                continue

            cand_health = self._health_by_key[key]
            similarity = float(1.0 - dist)

            # Prefer alternatives with better health score
            if cand_health < source_health - 5 and similarity < 0.85:
                continue

            reason = self._similarity_reason(profile, candidate, cand_health, source_health)
            candidates.append(
                SimilarFood(
                    food_name=candidate.food_name,
                    similarity=round(similarity * 100, 1),
                    calories=candidate.calories,
                    health_score=cand_health,
                    reason=reason,
                )
            )

        candidates.sort(key=lambda x: (-x.health_score, -x.similarity))
        return candidates[:top_k]

    def _similarity_reason(
        self,
        source: NutritionProfile,
        candidate: NutritionProfile,
        cand_health: float,
        source_health: float,
    ) -> str:
        parts = []
        if cand_health > source_health:
            parts.append(f"+{cand_health - source_health:.0f} health score")
        if candidate.calories < source.calories:
            parts.append("lower calories")
        if candidate.protein_g > source.protein_g:
            parts.append("more protein")
        if candidate.fiber_g > source.fiber_g:
            parts.append("more fiber")
        return ", ".join(parts) if parts else "similar macro profile"
