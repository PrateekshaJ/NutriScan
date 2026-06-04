import pandas as pd


class NutritionAnalyzer:

    def __init__(self):

        self.data = pd.read_csv(
            "data/nutrition_db.csv"
        )


        # remove useless columns
        self.data = self.data.loc[
            :,
            ~self.data.columns.str.contains(
                "Unnamed"
            )
        ]


        self.data.columns = (
            self.data.columns
            .str.lower()
            .str.strip()
        )


    def find_food(self, food_name):

        food_name = food_name.lower()


        results = self.data[
            self.data["food"]
            .str.lower()
            .str.contains(
                food_name,
                na=False
            )
        ]


        return results



    def calculate_health_score(
        self,
        food
    ):

        score = 100


        calories = food.get(
            "caloric_value",
            0
        )


        fat = food.get(
            "fat",
            0
        )


        sugar = food.get(
            "sugars",
            0
        )


        if calories > 300:
            score -= 20


        if fat > 15:
            score -= 20


        if sugar > 20:
            score -= 20


        return max(
            score,
            0
        )



    def recommendation(
        self,
        score
    ):

        if score >= 80:
            return "Excellent choice 🥦🔥"

        elif score >= 50:
            return "Balanced choice 👍"

        else:
            return "Eat occasionally 🍟"