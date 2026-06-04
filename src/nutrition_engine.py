import pandas as pd


class NutritionAnalyzer:

    def __init__(self):

        self.data = pd.read_csv(
            "data/nutrition_db.csv"
        )

        # remove csv index junk
        self.data = self.data.loc[
            :,
            ~self.data.columns.str.contains("^Unnamed")
        ]

        self.data.columns = (
            self.data.columns
            .str.lower()
            .str.strip()
        )


    def find_food(self, food_name):

        food_name = food_name.lower()

        result = self.data[
            self.data["food"]
            .str.lower()
            .str.contains(food_name, na=False)
        ]

        if len(result) == 0:
            return None

        return result.iloc[0]


    def health_score(self, food):

        score = 100


        calories = food.get(
            "caloric_value",
            0
        )

        sugar = food.get(
            "sugars",
            0
        )

        fat = food.get(
            "fat",
            0
        )

        protein = food.get(
            "protein",
            0
        )


        # simple AI scoring logic
        if calories > 500:
            score -= 20

        if sugar > 30:
            score -= 20

        if fat > 20:
            score -= 15

        if protein > 10:
            score += 10


        score = max(
            0,
            min(
                100,
                score
            )
        )

        return round(
            score,
            2
        )


    def recommendation(self, score):

        if score >= 80:

            return "Excellent choice 🥦🔥"

        elif score >= 50:

            return "Balanced choice 👍"

        else:

            return "Consume occasionally 🍟"