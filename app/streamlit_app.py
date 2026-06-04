import streamlit as st
import sys
import os
from PIL import Image


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src.nutrition_engine import NutritionAnalyzer
from src.food_detector import FoodDetector


st.set_page_config(
    page_title="NutriScan",
    page_icon="🥗",
    layout="wide"
)


engine = NutritionAnalyzer()
detector = FoodDetector()


st.title("NutriScan 🥗")

st.write(
    "AI-powered food nutrition analyzer"
)


# IMPORTANT
selected = None


tab1, tab2 = st.tabs(
    [
        "🔍 Search Food",
        "📸 Scan Food"
    ]
)


# ---------------- SEARCH FOOD ---------------- #

with tab1:

    food = st.text_input(
        "Search your food 🍎"
    )


    if food:

        results = engine.find_food(
            food
        )


        if len(results) == 0:

            st.error(
                "Food not found 😭"
            )

        else:

            selected = results.iloc[0]



# ---------------- IMAGE DETECTION ---------------- #

with tab2:

    image_file = st.file_uploader(
        "Upload food image 📸",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if image_file:

        img = Image.open(
            image_file
        ).convert(
            "RGB"
        )


        st.image(
            img,
            width=300
        )


        food_name, confidence = detector.predict(
            img
        )


        st.success(
            f"Detected: {food_name} 🥗 ({confidence}%)"
        )


        food_name = food_name.replace(
            "_",
            " "
        )


        results = engine.find_food(
            food_name
        )


        if len(results) == 0:

            st.warning(
                "Nutrition data unavailable 😭"
            )

        else:

            selected = results.iloc[0]



# ---------------- SHOW RESULT ---------------- #

if selected is not None:


    st.divider()


    st.subheader(
        selected["food"]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Protein 💪",
            f'{selected.get("protein",0)} g'
        )


    with col2:

        st.metric(
            "Fat 🧈",
            f'{selected.get("fat",0)} g'
        )


    with col3:

        st.metric(
            "Calories 🔥",
            selected.get(
                "caloric_value",
                0
            )
        )


    score = engine.calculate_health_score(
        selected
    )


    st.metric(
        "Health Score 🥗",
        f"{score}/100"
    )


    st.success(
        engine.recommendation(
            score
        )
    )


    st.divider()


    st.subheader(
        "Complete Nutrition Data"
    )


    st.dataframe(
        selected
    )