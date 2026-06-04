from src.nutrition_engine import NutritionAnalyzer


engine = NutritionAnalyzer()


food = engine.find_food(
    "rice"
)


print(food)


if len(food)>0:

    selected = food.iloc[0]

    print(
        "Health Score:",
        engine.health_score(selected)
    )

    print(
        engine.recommendation(
            engine.health_score(selected)
        )
    )