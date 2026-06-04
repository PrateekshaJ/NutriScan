import streamlit as st
import sys
import os


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src.nutrition_engine import NutritionAnalyzer


st.set_page_config(
    page_title="NutriScan",
    page_icon="🥗",
    layout="wide"
)


engine = NutritionAnalyzer()


st.title("NutriScan 🥗")

st.write(
    "AI-powered food nutrition analyzer"
)


food = st.text_input(
    "Search your food 🍎"
)


if food:

    selected = engine.find_food(food)


    if selected is None:

        st.error(
            "Food not found 😭"
        )


    else:

        st.subheader(
            selected["food"]
        )


        col1, col2, col3 = st.columns(3)


        with col1:
            st.metric(
                "Protein 💪",
                f"{selected.get('protein',0):.2f} g"
            )


        with col2:
            st.metric(
                "Fat 🧈",
                f"{selected.get('fat',0):.2f} g"
            )


        with col3:
            st.metric(
                "Calories 🔥",
                f"{selected.get('caloric_value',0):.0f}"
            )


        score = engine.health_score(
            selected
        )


        st.metric(
            "Health Score 🥗",
            f"{score}/100"
        )


        message = engine.recommendation(
            score
        )


        if score >= 80:
            st.success(message)

        elif score >= 50:
            st.info(message)

        else:
            st.warning(message)


        st.divider()


        st.subheader(
            "Complete Nutrition Data"
        )


        clean_data = selected.drop(
            labels=[
                x for x in selected.index
                if "unnamed" in x.lower()
            ],
            errors="ignore"
        )


        st.dataframe(
            clean_data.to_frame("value")
        )