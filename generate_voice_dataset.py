# ================================
# Generate Campus Voice Dataset
# 60 WAV files for Whisper Testing
# ================================

import os
from gtts import gTTS
from pydub import AudioSegment


# ----------------------------
# Create folders
# ----------------------------

BASE_PATH = "dataset/audio"

folders = [
    "find_location",
    "ask_hours",
    "find_event",
    "ask_direction"
]


for folder in folders:
    os.makedirs(
        os.path.join(BASE_PATH, folder),
        exist_ok=True
    )


# ----------------------------
# Voice query dataset
# ----------------------------

voice_queries = {


"find_location": [

"Where is the library?",
"Find the computer room.",
"Where can I find the cafeteria?",
"Show me the auditorium location.",
"Where is the classroom?",
"Find the laboratory.",
"Where is the gym?",
"Locate the meeting room.",
"Where is the student lobby?",
"Find the bookstore.",
"Where is the office?",
"Show me the museum location.",
"Where is the music studio?",
"Find the art studio.",
"Where is the dining room?",
"Locate the waiting room.",
"Where is the elevator?",
"Find the locker room.",
"Where is the staircase?",
"Show campus corridor location."

],


"ask_hours": [

"What time does the library open?",
"When does the cafeteria close?",
"Computer room opening hours.",
"What are laboratory working hours?",
"Tell me office opening time.",
"When is the gym available?",
"Auditorium opening hours please.",
"What time does the bookstore open?",
"Meeting room availability time.",
"When does the museum close?",
"Dining room working hours.",
"Student lobby opening hours.",
"What is the schedule of art studio?",
"Music studio opening time.",
"Reception working hours."

],



"find_event": [

"What events are happening today?",
"Show upcoming campus events.",
"Any workshop in the library?",
"What activities are in auditorium?",
"Show student programs today.",
"Any fitness event in gym?",
"Find academic seminars.",
"What exhibition is available?",
"Show music studio events.",
"Any creative workshop today?"

],



"ask_direction": [

"How can I reach the library?",
"Give directions to cafeteria.",
"Navigate me to computer room.",
"How do I go to classroom?",
"Show route to laboratory.",
"Directions to auditorium please.",
"How can I reach gym?",
"Navigate me to office.",
"Show path to meeting room.",
"How do I find bookstore?",
"Guide me to museum.",
"Where should I go for dining room?",
"Show direction to elevator.",
"Guide me to music studio.",
"Navigate to student lobby."

]

}


# ----------------------------
# Generate WAV files
# ----------------------------


for intent, queries in voice_queries.items():

    for index, text in enumerate(queries):

        filename = f"{intent}_{index+1}.wav"

        save_path = os.path.join(
            BASE_PATH,
            intent,
            filename
        )


        # temporary mp3
        temp_mp3 = "temp.mp3"


        # text to speech
        tts = gTTS(
            text=text,
            lang="en"
        )

        tts.save(temp_mp3)


        # convert MP3 to WAV
        audio = AudioSegment.from_mp3(
            temp_mp3
        )

        audio.export(
            save_path,
            format="wav"
        )


        os.remove(temp_mp3)

        print(
            "Created:",
            save_path
        )


print(
    "\nVoice Dataset Completed Successfully!"
)
