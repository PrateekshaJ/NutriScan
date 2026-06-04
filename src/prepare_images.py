import os
import shutil
import random


SOURCE = r"C:\Users\DELL\Downloads\archive (5)\food-101\food-101\images"

DEST = "food_images"


classes = [
    "apple_pie",
    "pizza",
    "hamburger",
    "french_fries",
    "ice_cream",
    "omelette",
    "pancakes",
    "club_sandwich",
    "spaghetti_bolognese",
    "sushi",
    "chocolate_cake",
    "donuts",
    "fried_rice",
    "hot_dog",
    "tacos"
]


for food in classes:

    images = os.listdir(
        os.path.join(
            SOURCE,
            food
        )
    )


    random.shuffle(images)


    train = images[:700]
    val = images[700:900]


    for folder_name, files in [
        ("train", train),
        ("val", val)
    ]:

        path = os.path.join(
            DEST,
            folder_name,
            food
        )

        os.makedirs(
            path,
            exist_ok=True
        )


        for img in files:

            shutil.copy(
                os.path.join(
                    SOURCE,
                    food,
                    img
                ),
                os.path.join(
                    path,
                    img
                )
            )


print("Food image dataset ready 🥗🔥")