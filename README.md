# Smart Campus Multimodal AI Assistant

A proof-of-concept multimodal chatbot for university campus orientation. The system processes three input modalities — images, voice, and text — to help students and visitors find campus locations, check opening hours, discover events, and get directions.

Built as part of the BSBI campus orientation assignment demonstrating the complete multimodal AI pipeline: data acquisition, preprocessing, model design, multimodal fusion, training, evaluation, deployment, and ethical considerations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Technology Stack](#technology-stack)
5. [Environment Setup](#environment-setup)
6. [Dataset Description](#dataset-description)
7. [Model Pipeline](#model-pipeline)
8. [Preprocessing](#preprocessing)
9. [Training and Evaluation](#training-and-evaluation)
10. [Streamlit Application](#streamlit-application)
11. [Docker Deployment](#docker-deployment)
12. [Test Scenarios](#test-scenarios)
13. [Evaluation Metrics](#evaluation-metrics)
14. [Ethical and Regulatory Considerations](#ethical-and-regulatory-considerations)

---

## Project Overview

The Smart Campus Multimodal AI Assistant accepts user input through three modalities:

- **Image Upload** — Upload a photo of a campus building or sign. The system uses CLIP embeddings and FAISS vector search to identify the closest matching campus location.
- **Voice Input** — Upload an audio recording of a spoken question. Whisper transcribes the audio to text, then DistilBERT classifies the intent and retrieves the answer from the knowledge base.
- **Text Input** — Type a question directly. DistilBERT classifies the intent and the system returns relevant campus information.

The system returns the matched building name, description, opening hours, directions (map reference), and upcoming events from a structured campus knowledge base.

---

## Architecture

```
User Input (Image / Voice / Text)
        |
        v
+------------------+     +------------------+     +------------------+
|   CLIP Encoder   |     |   Whisper ASR    |     |   Text Input     |
| (ViT-B/32)       |     |   (base model)   |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                         |
         v                        v                         v
+------------------+     +------------------+     +------------------+
|   FAISS Index    |     |   Transcript     |     |   DistilBERT     |
| (Cosine Search)  |     |   (ASR Output)   |     | (Intent Classifier)|
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                         |
         v                        +----------+--------------+
         |                                   |
         v                                   v
+------------------+              +------------------+
| Multimodal       |              | Intent + Entity  |
| Fusion MLP       |              | Extraction       |
+--------+---------+              +--------+---------+
         |                                   |
         +----------------+------------------+
                          |
                          v
              +---------------------+
              | Campus Knowledge    |
              | Base (JSON, 20 rec) |
              +----------+----------+
                         |
                         v
              +---------------------+
              | Response Generation |
              | (Streamlit UI)      |
              +---------------------+
```

---

## Project Structure

```
Smart_Campus_Multimodal_AI_Assistant/
|
|-- app.py                          Main Streamlit web application
|-- requirements.txt                Python dependencies
|-- Dockerfile                      Docker container configuration
|-- .dockerignore                   Files excluded from Docker build
|-- README.md                       Project documentation (this file)
|-- generate_faq_dataset.py         Script to generate synthetic FAQ queries
|-- generate_voice_dataset.py       Script to generate synthetic voice audio
|
|-- dataset/
|   |-- images/                     Campus location image dataset (19 classes)
|   |   |-- artstudio/              Creative Art Studio images (140 images)
|   |   |-- auditorium/             Auditorium Hall images (176 images)
|   |   |-- bookstore/              Campus Bookstore images (380 images)
|   |   |-- classroom/              Lecture Classroom images (113 images)
|   |   |-- corridor/               Campus Corridor images (346 images)
|   |   |-- dining_room/            Dining Area images (274 images)
|   |   |-- elevator/               Elevator Area images (101 images)
|   |   |-- gym/                    Sports Centre images (231 images)
|   |   |-- laboratorywet/          Science Laboratory images (125 images)
|   |   |-- library/                Main Library images (107 images)
|   |   |-- lobby/                  Main Reception images (101 images)
|   |   |-- locker_room/            Student Locker Room images (249 images)
|   |   |-- meeting_room/           Meeting Room images (233 images)
|   |   |-- museum/                 Campus Museum images (168 images)
|   |   |-- office/                 Administration Office images (109 images)
|   |   |-- restaurant_kitchen/     Campus Cafeteria images (107 images)
|   |   |-- stairscase/             Staircase Area images (155 images)
|   |   |-- studiomusic/            Music Studio images (108 images)
|   |   |-- waitingroom/            Student Waiting Area images (151 images)
|   |
|   |-- audio/                      Synthetic voice queries (4 intent categories)
|   |   |-- ask_direction/          Direction-related voice queries (20 files)
|   |   |-- ask_hours/              Opening hours voice queries (20 files)
|   |   |-- find_event/             Event-related voice queries (20 files)
|   |   |-- find_location/          Location-finding voice queries (20 files)
|   |
|   |-- text/
|   |   |-- campus_queries.csv      200 labelled text queries (query, intent, location)
|   |
|   |-- knowledge_base/
|       |-- campus_database.json    Structured campus KB with 20 location records
|
|-- saved_models/
|   |-- distilbert/                 Fine-tuned DistilBERT intent classifier
|   |   |-- config.json             Model configuration (4 intent classes)
|   |   |-- model.safetensors       Model weights
|   |   |-- tokenizer.json          Tokenizer data
|   |   |-- tokenizer_config.json   Tokenizer configuration
|   |   |-- checkpoint-10/          Training checkpoint (epoch 1)
|   |   |-- checkpoint-20/          Training checkpoint (epoch 2)
|   |   |-- checkpoint-30/          Training checkpoint (epoch 3)
|   |   |-- checkpoint-40/          Training checkpoint (epoch 4)
|   |   |-- checkpoint-50/          Training checkpoint (epoch 5)
|   |
|   |-- faiss_index/
|   |   |-- campus.index            FAISS index with 285 CLIP image embeddings
|   |   |-- location_mapping.pkl    Label mapping for FAISS index entries
|   |
|   |-- fusion/
|       |-- fusion_model.pt         Trained Fusion MLP weights
|
|-- venv/                           Python virtual environment (not in Docker)
```

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Deep Learning | PyTorch, TorchVision | Model training and inference |
| Vision Model | CLIP (openai/clip-vit-base-patch32) | Image embedding extraction |
| Vector Search | FAISS (faiss-cpu) | Cosine similarity search on image embeddings |
| Speech-to-Text | OpenAI Whisper (base) | Audio transcription |
| NLP Model | DistilBERT (fine-tuned) | Intent classification (4 classes) |
| Fusion | Custom MLP (PyTorch) | Multimodal embedding fusion |
| Image Processing | OpenCV, Pillow | Image loading and preprocessing |
| Audio Processing | Librosa, SoundFile | Audio feature extraction (MFCCs) |
| Web UI | Streamlit | Interactive web application |
| Containerisation | Docker | Application deployment |
| Data Handling | Pandas, NumPy | Dataset management |
| Evaluation | scikit-learn, jiwer | Metrics computation (accuracy, F1, WER) |
| Visualisation | Matplotlib | Training curves and data exploration plots |

---

## Environment Setup

### Prerequisites

- Python 3.10 or 3.11
- pip package manager
- Docker (for containerised deployment)

### Local Installation

```bash
git clone <repository-url>
cd Smart_Campus_Multimodal_AI_Assistant

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Dependency List

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
openai-whisper>=20230918
opencv-python-headless>=4.8.0
Pillow>=10.0.0
librosa>=0.10.0
soundfile>=0.12.0
faiss-cpu>=1.7.4
jiwer>=3.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
streamlit>=1.28.0
accelerate>=0.20.0
```

---

## Dataset Description

### Visual Data (3,374 images across 19 classes)

Images are sourced from indoor scene recognition datasets and organised into 19 campus location categories. Each folder contains between 101 and 380 images. The dataset was split 80/20 for training and validation (2,699 train / 675 val).

### Voice Data (80 synthetic audio files)

Generated using text-to-speech tools across 4 intent categories with 20 WAV files each:
- **ask_direction** — e.g., "How can I reach the library?"
- **ask_hours** — e.g., "What time does the cafeteria close?"
- **find_event** — e.g., "What events are happening today?"
- **find_location** — e.g., "Where is the library?"

### Text Data (200 labelled queries)

A CSV file (`campus_queries.csv`) containing 200 queries with columns: `query`, `intent`, and `location`. Distributed evenly across 4 intent categories. Split 80/20 for training and validation (160 train / 40 val).

### Campus Knowledge Base (20 location records)

A JSON file (`campus_database.json`) with structured records for each campus location. Each record contains:
- `id` — Unique identifier
- `building_name` — Display name (e.g., "Main Library")
- `folder_name` — Corresponding image folder name
- `category` — Location type (e.g., "Library", "Lecture Hall")
- `description` — Short description of the facility
- `opening_hours` — Operating hours
- `map_reference` — Relative campus location / directions
- `events` — Upcoming events at the location

The knowledge base is queried at inference time based on the detected intent and extracted entity. The system matches user queries against building names, categories, and folder names to retrieve the correct record.

---

## Model Pipeline

### 1. CLIP + FAISS (Vision)

- Model: `openai/clip-vit-base-patch32` (frozen, no training)
- 285 image embeddings (768-dimensional) indexed in FAISS using cosine similarity
- At query time, the uploaded image is encoded by CLIP and the nearest neighbours are retrieved from the FAISS index
- Returns Top-1 and Top-3 matched locations

### 2. Whisper (Speech-to-Text)

- Model: `whisper-base` (frozen, no training)
- Transcribes uploaded audio files to text
- The transcribed text is then passed to DistilBERT for intent classification

### 3. DistilBERT (Intent Classification)

- Model: `distilbert-base-uncased`, fine-tuned on 200 campus queries
- 4 intent classes: `ask_direction`, `ask_hours`, `find_event`, `find_location`
- Trained for 5 epochs with batch size 16, learning rate warmup, and weight decay
- Best model selected based on validation accuracy (100% at epoch 4)

### 4. Fusion MLP (Multimodal Fusion)

Custom PyTorch MLP combining image and text semantic embeddings.

Architecture:
projection layers (256 dimensions each)
→ embedding concatenation
→ fully connected MLP layers with ReLU activation and dropout
→ 4-class intent output layer

The fusion model predicts:
- ask_direction
- ask_hours
- find_event
- find_location

Missing modalities are handled using zero-vector padding, allowing the system to process image-only, voice-only, text-only, and combined inputs.

### 5. Knowledge Base Retrieval

- Structured JSON lookup matching detected intent and extracted entity against building records
- Entity extraction matches user text against known building names, folder names, and categories
- Fallback logic returns event-having records for event queries, or the first record as default

---

## Preprocessing

### Image Pipeline
- Resize to 224x224 pixels
- Random rotation (15 degrees), horizontal flip, and colour jitter for augmentation
- Convert to tensor and normalise with ImageNet mean/std values
- Batch loading with DataLoader (batch size 16)

### Audio Pipeline
- Load audio with Librosa at native sample rate
- Extract 13 MFCC features for analysis and visualisation
- Whisper handles its own internal preprocessing for transcription

### Text Pipeline
- Lowercase conversion
- Remove special characters and punctuation
- Remove stopwords (a, an, the, is, in, at, of, to, and, or, for, on, are, was, it, with)
- Tokenisation with DistilBERT tokenizer (max length 64)
- Train/validation split with stratification (80/20)

---

## Training and Evaluation

### DistilBERT Training

| Epoch | Training Loss | Validation Loss | Accuracy |
|---|---|---|---|
| 1 | 1.3780 | 1.3480 | 57.5% |
| 2 | 1.2318 | 1.0071 | 67.5% |
| 3 | 0.7289 | 0.4434 | 97.5% |
| 4 | 0.3448 | 0.2067 | 100.0% |
| 5 | 0.1911 | 0.1492 | 100.0% |

### Fusion MLP Training

| Epoch | Training Loss | Validation Accuracy |
|---|---|---|
| 1 | 2.5518 | 30.0% |
| 2 | 1.5869 | 25.0% |
| 3 | 1.4420 | 45.0% |
| 4 | 1.4497 | 30.0% |
| 5 | 1.4316 | 45.0% |
| 6 | 1.3658 | 67.5% |
| 7 | 1.3320 | 45.0% |
| 8 | 1.2875 | 45.0% |
| 9 | 1.2490 | 47.5% |
| 10 | 1.1985 | 47.5% |

---

## Streamlit Application

### Running the Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

### Interface

The application has three input tabs:

**Image Input Tab**
- Upload a campus photo (JPG, JPEG, PNG, BMP, WEBP)
- Displays the uploaded image alongside Top-3 retrieval matches with similarity scores
- Shows the matched location details from the knowledge base

**Voice Input Tab**
- Upload an audio file (WAV, MP3, M4A, OGG, FLAC)
- Plays the audio in the browser
- Displays the Whisper transcript
- Shows the classified intent and matched knowledge base record

**Text Input Tab**
- Type a question or click an example query button
- Displays the classified intent with confidence score
- Shows the matched knowledge base record

**Knowledge Base Browser**
- Expandable section at the bottom showing all 20 campus locations in a table

---

## Docker Deployment

### Build the Docker Image

```bash
docker build -t smart-campus-assistant .
```

### Run the Container

```bash
docker run -p 8501:8501 smart-campus-assistant
```

### Access the Application

Open `http://localhost:8501` in a web browser.

### Dockerfile Details

- **Base image:** `python:3.11-slim`
- **System dependencies:** FFmpeg (required by Whisper), libsndfile (audio processing), OpenGL and GLib (OpenCV)
- **Layer caching:** `requirements.txt` is copied and installed before the application code to optimise rebuild times
- **Health check:** Polls the Streamlit health endpoint every 30 seconds
- **Entrypoint:** Runs Streamlit in headless mode on port 8501

### .dockerignore

The following are excluded from the Docker build context:
- `venv/` (virtual environment)
- `__pycache__/` and `.pyc` files
- Training checkpoints (`checkpoint-*/`)
- `.git/` directory
- `.zip` files

---

## Test Scenarios

| Scenario | Input Type | Query / File | Expected Result | Actual Result |
|---|---|---|---|---|
| 1 | Text | "Where is the library?" | Main Library | Main Library (find_location) |
| 2 | Voice | find_location_1.wav ("Where is the library?") | Main Library | Main Library (find_location) |
| 3 | Text | "What time does the cafeteria open?" | Campus Cafeteria | Campus Cafeteria (ask_hours) |
| 4 | Text | "Are there events at the auditorium?" | Auditorium Hall | Auditorium Hall (find_event) |
| 5 | Text | "How do I get to the gym?" | Sports Centre | Sports Centre (ask_direction) |

---

## Evaluation Metrics

| Metric | Value |
|---|---|
| CLIP Top-1 Retrieval Accuracy | 81.1% (77/95) |
| CLIP Top-3 Retrieval Accuracy | 95.8% (91/95) |
| DistilBERT Intent Accuracy | 100% (40/40) |
| DistilBERT Precision (macro) | 1.00 |
| DistilBERT Recall (macro) | 1.00 |
| DistilBERT F1 Score (macro) | 1.00 |
| Whisper Word Error Rate (WER) | 243.75% |
| End-to-End KB Retrieval Accuracy | 60.0% (3/5) |

