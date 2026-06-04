"""
NutriScan — AI-powered food intelligence dashboard.

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path when launched via streamlit run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.food_detector import FoodDetector
from src.health_score import HealthScoreEngine
from src.nutrition_engine import NutritionEngine
from src.recommender import FoodRecommender, GoalType

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NutriScan | Food Intelligence",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #134e4a 50%, #042f2e 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.4rem 0 0; opacity: 0.9; font-size: 1rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    .score-ring { font-size: 2.5rem; font-weight: 700; color: #0f766e; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_services():
    """Initialize ML services once per session."""
    detector = FoodDetector()
    nutrition = NutritionEngine()
    health = HealthScoreEngine()
    recommender = FoodRecommender(nutrition_engine=nutrition, health_engine=health)
    return detector, nutrition, health, recommender


def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🥗 NutriScan</h1>
            <p>AI-powered food intelligence & nutrition analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_card(result, demo_mode: bool):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Detected Food", result.food_name)
        st.metric("Confidence", f"{result.confidence_percent}%")
        if demo_mode:
            st.caption("Demo classifier — train a model with train_model.py for production.")
    with col2:
        st.subheader("Top predictions")
        labels = [p["food"] for p in result.top_predictions]
        scores = [p["confidence"] * 100 for p in result.top_predictions]
        fig = px.bar(
            x=scores,
            y=labels,
            orientation="h",
            labels={"x": "Confidence %", "y": ""},
            color=scores,
            color_continuous_scale="Teal",
        )
        fig.update_layout(
            height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_nutrition_metrics(profile):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", f"{profile.calories:.0f} kcal")
    c2.metric("Protein", f"{profile.protein_g:.1f} g")
    c3.metric("Carbs", f"{profile.carbohydrates_g:.1f} g")
    c4.metric("Fats", f"{profile.fats_g:.1f} g")


def render_macro_chart(profile):
    macro_df = {
        "Nutrient": ["Protein", "Carbohydrates", "Fats"],
        "Grams": [profile.protein_g, profile.carbohydrates_g, profile.fats_g],
    }
    fig = px.pie(
        macro_df,
        values="Grams",
        names="Nutrient",
        hole=0.45,
        color_discrete_sequence=["#0d9488", "#14b8a6", "#5eead4"],
    )
    fig.update_layout(height=320, margin=dict(t=20, b=20))
    return fig


def render_health_score(health_result):
    col_score, col_breakdown = st.columns([1, 2])
    with col_score:
        st.markdown(
            f'<p class="score-ring">{health_result.overall_score}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{health_result.grade}** · Health Score (0–100)")
        for insight in health_result.insights:
            st.info(insight)

    with col_breakdown:
        sub = {
            "Dimension": ["Calories", "Protein", "Fat", "Balance"],
            "Score": [
                health_result.calorie_score,
                health_result.protein_score,
                health_result.fat_score,
                health_result.balance_score,
            ],
        }
        fig = go.Figure(
            data=go.Scatterpolar(
                r=sub["Score"] + [sub["Score"][0]],
                theta=sub["Dimension"] + [sub["Dimension"][0]],
                fill="toself",
                line_color="#0f766e",
                fillcolor="rgba(15, 118, 110, 0.3)",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            height=320,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_recommendations(bundle):
    st.subheader("Goal assessment")
    assessment = bundle.assessment
    icon = "✅" if assessment.fits_goal else "⚠️"
    st.markdown(f"{icon} {assessment.verdict}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Improvements**")
        for item in assessment.improvements:
            st.markdown(f"- {item}")
    with col_b:
        st.markdown("**Healthier swaps**")
        for swap in assessment.healthier_swaps:
            st.markdown(f"- {swap}")

    st.subheader("Similar healthier alternatives")
    if not bundle.similar_foods:
        st.caption("No similar alternatives found in the nutrition database.")
        return

    for sim in bundle.similar_foods:
        with st.container():
            st.markdown(
                f"**{sim.food_name}** · {sim.similarity}% similar · "
                f"{sim.calories:.0f} kcal · Health {sim.health_score:.0f}/100"
            )
            st.caption(sim.reason)


def main():
    render_header()
    detector, nutrition, health_engine, recommender = load_services()

    with st.sidebar:
        st.header("Settings")
        goal_label = st.selectbox(
            "Your goal",
            [g.value for g in GoalType],
            index=2,
        )
        goal = GoalType(goal_label)
        manual_override = st.checkbox("Manual food override", value=False)
        manual_food = None
        if manual_override:
            manual_food = st.selectbox(
                "Select food",
                nutrition.available_foods,
            )
        st.divider()
        st.markdown("**About**")
        st.caption(
            "Upload a food photo for AI detection, nutrition facts, "
            "health scoring, and personalized recommendations."
        )

    uploaded = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear, well-lit photos improve recognition accuracy.",
    )

    if uploaded is None and not manual_override:
        st.info("Upload an image or enable manual food override in the sidebar to begin.")
        return

    with st.spinner("Analyzing food..."):
        if uploaded is not None:
            image_bytes = uploaded.read()
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)
            detection = detector.predict(image_bytes)
        else:
            detection = None

        food_key = (
            manual_food.lower().replace(" ", "_")
            if manual_food
            else nutrition._normalize_key(detection.food_name)
        )
        profile = nutrition.get_profile(food_key)
        summary = nutrition.build_summary(profile)
        health_result = health_engine.score(profile)
        bundle = recommender.recommend(profile, goal, health_result)

    tab_detect, tab_nutrition, tab_health, tab_ai = st.tabs(
        ["Detection", "Nutrition", "Health Score", "AI Insights"]
    )

    with tab_detect:
        if detection:
            render_detection_card(detection, detector.is_demo_mode)
        else:
            st.success(f"Manual selection: **{profile.food_name}**")

    with tab_nutrition:
        st.subheader(profile.food_name)
        render_nutrition_metrics(profile)
        st.markdown(summary)
        col_chart, col_extra = st.columns(2)
        with col_chart:
            st.plotly_chart(render_macro_chart(profile), use_container_width=True)
        with col_extra:
            st.markdown("**Additional nutrients**")
            st.write(f"Fiber: **{profile.fiber_g:.1f} g**")
            st.write(f"Sugar: **{profile.sugar_g:.1f} g**")
            st.write(f"Sodium: **{profile.sodium_mg:.0f} mg**")
            st.write(f"Serving: **{profile.serving_size_g:.0f} g**")

    with tab_health:
        render_health_score(health_result)

    with tab_ai:
        render_recommendations(bundle)


if __name__ == "__main__":
    main()
