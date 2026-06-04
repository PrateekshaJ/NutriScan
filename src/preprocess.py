import pandas as pd
import os


RAW_PATH = "data"
OUTPUT_PATH = "data/nutrition_db.csv"


def combine_food_data():

    all_data = []

    for file in os.listdir(RAW_PATH):

        if file.endswith(".csv") and file != "nutrition_db.csv":

            path = os.path.join(RAW_PATH, file)

            print("Loading:", file)

            df = pd.read_csv(path)

            all_data.append(df)


    food_df = pd.concat(
        all_data,
        ignore_index=True
    )


    # clean column names
    food_df.columns = (
        food_df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )


    # remove duplicate foods
    food_df.drop_duplicates(
        inplace=True
    )


    # fill missing values
    food_df.fillna(
        0,
        inplace=True
    )


    food_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "Dataset created:",
        food_df.shape
    )



if __name__ == "__main__":
    combine_food_data()