# ======================================
# Generate Campus FAQ Dataset
# For DistilBERT Intent Classification
# 200 Questions
# ======================================

import pandas as pd
import os
import random


SAVE_PATH = "dataset/text"

os.makedirs(
    SAVE_PATH,
    exist_ok=True
)


locations = [

"library",
"classroom",
"computer room",
"laboratory",
"office",
"corridor",
"lobby",
"meeting room",
"auditorium",
"cafeteria",
"bookstore",
"gym",
"museum",
"staircase",
"elevator",
"locker room",
"dining room",
"waiting room",
"art studio",
"music studio"

]


data = []


# -------------------------------
# find_location - 60 samples
# -------------------------------

templates = [

"Where is the {}?",
"Find the {}.",
"Can you locate the {}?",
"I want to find the {}.",
"Show me the location of {}.",
"Where can I access the {}?"

]

for i in range(60):

    loc = random.choice(locations)

    query = random.choice(
        templates
    ).format(loc)

    data.append(
        [
        query,
        "find_location",
        loc
        ]
    )



# -------------------------------
# ask_hours - 50 samples
# -------------------------------

templates = [

"What time does the {} open?",
"When does the {} close?",
"Tell me opening hours of {}.",
"What are the working hours for {}?",
"Is the {} open today?"

]


for i in range(50):

    loc=random.choice(
        locations
    )

    query=random.choice(
        templates
    ).format(loc)

    data.append(
        [
        query,
        "ask_hours",
        loc
        ]
    )



# -------------------------------
# ask_direction - 50 samples
# -------------------------------

templates = [

"How can I reach the {}?",
"Give me directions to the {}.",
"Navigate me to {}.",
"Show the route to {}.",
"How do I get to the {}?"

]


for i in range(50):

    loc=random.choice(
        locations
    )

    query=random.choice(
        templates
    ).format(loc)


    data.append(
        [
        query,
        "ask_direction",
        loc
        ]
    )



# -------------------------------
# find_event - 40 samples
# -------------------------------


templates = [

"What events are happening in the {}?",
"Show upcoming events at {}.",
"Any activities in the {}?",
"Are there workshops at {}?",
"Find programs happening in {}."

]


for i in range(40):

    loc=random.choice(
        locations
    )

    query=random.choice(
        templates
    ).format(loc)


    data.append(
        [
        query,
        "find_event",
        loc
        ]
    )


# Shuffle dataset

random.shuffle(data)


# Create dataframe

df=pd.DataFrame(
    data,
    columns=[
        "query",
        "intent",
        "location"
    ]
)



# Save CSV

df.to_csv(
    SAVE_PATH+
    "/campus_queries.csv",
    index=False
)


print(
    "FAQ Dataset Created Successfully"
)

print(
    df.head()
)

print(
    "\nTotal Records:",
    len(df)
)

print(
    df["intent"].value_counts()
)
