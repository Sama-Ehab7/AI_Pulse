# AI_Pulse
# 🧠 Aspect-Based Sentiment Analysis (ABSA) System

## 📌 Overview

This project presents a **production-ready Aspect-Based Sentiment Analysis (ABSA)** system designed to process **real-world Arabic customer reviews**. The system is capable of identifying multiple aspects within a single review and assigning a sentiment to each aspect independently.

Unlike traditional sentiment classification, this solution captures **fine-grained insights** by modeling both:

* **Aspect extraction (multi-label classification)**
* **Aspect-level sentiment prediction (multi-class classification)**

The system is built to generalize across multiple domains, including restaurants, e-commerce, healthcare, and mobile applications.

---

## 🎯 Objectives

The primary objectives of this project are:

1. **Aspect Detection**
   Identify all relevant aspects mentioned in a review.

2. **Aspect-Level Sentiment Classification**
   Assign one sentiment label (positive, negative, neutral) to each detected aspect.

3. **Robustness to Noisy Data**
   Handle informal Arabic, mixed-language content, and real-world user-generated text.

4. **Production-Ready Output**
   Generate predictions in a strict JSON format required for evaluation.

---

## 🔍 Supported Aspect Taxonomy

The model strictly adheres to the predefined aspect categories:

* `food`
* `service`
* `price`
* `cleanliness`
* `delivery`
* `ambiance`
* `app_experience`
* `general`
* `none`

Only these aspect labels are used to ensure compatibility with the evaluation system.

---

## 🏗️ System Architecture

The solution follows a **two-stage modular pipeline**:

### 1. Aspect Detection Model

* **Input:** `review_text`
* **Output:** Multi-label vector representing detected aspects

### 2. Aspect-Level Sentiment Model

* **Input:** (`review_text`, `aspect`)
* **Output:** Sentiment label (`positive`, `negative`, `neutral`)

This modular design improves flexibility, interpretability, and scalability.

---

## 📂 Project Structure

```bash
project/
│
├── data/                  # Dataset references (excluded from submission)
├── models/                # Saved trained model weights
├── notebooks/             # Experimentation and training notebooks
├── src/
│   ├── preprocessing.py   # Text cleaning and encoding
│   ├── train.py           # Model training pipeline
│   ├── inference.py       # Prediction generation
│   ├── utils.py           # Utility functions
│
├── outputs/
│   └── predictions.json   # Final submission file
│
├── requirements.txt
├── README.md
└── main.py / run.sh
```

---

## ⚙️ Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Training

```bash
python src/train.py
```

This step will:

* Load and preprocess the training data
* Train both aspect and sentiment models
* Save trained weights in the `/models` directory

---

### 2. Inference

```bash
python src/inference.py
```

This step will:

* Load trained models
* Run predictions on the hidden test dataset
* Generate the final `predictions.json` file

---

## 📊 Input Format

Each input sample contains:

| Field       | Description              |
| ----------- | ------------------------ |
| review_id   | Unique identifier        |
| review_text | Raw customer review text |

---

## 📤 Output Format (Submission)

The output must strictly follow this JSON schema:

```json
[
  {
    "review_id": 23,
    "aspects": ["service", "food"],
    "aspect_sentiments": {
      "service": "positive",
      "food": "negative"
    }
  }
]
```

---

## ⚠️ Validation Rules

To ensure correct evaluation:

* All `review_id` values must be present (no missing entries)
* `aspects` must only contain valid labels from the taxonomy
* Each aspect must appear exactly once in `aspect_sentiments`
* Sentiment values must be one of:

  * `positive`
  * `negative`
  * `neutral`

Any mismatch will negatively impact the final F1 score.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Arabic text normalization (handling variations in characters)
* Noise removal (punctuation, symbols, etc.)
* Multi-label encoding for aspect detection
* Label encoding for sentiment classification

The system is designed to handle:

* Noisy user-generated text
* Mixed Arabic-English content
* Informal writing styles

---

## 🤖 Model Details

* **Base Model:** Pretrained Arabic Transformer (e.g., AraBERT)
* **Tasks:**

  * Multi-label classification for aspects
  * Multi-class classification for sentiment

The model leverages transfer learning to achieve strong performance on limited labeled data.

---

## 📈 Evaluation

* **Metric:** F1 Score (Micro)
* Evaluation is performed on a hidden labeled test set

Performance depends on:

* Accurate aspect extraction
* Correct sentiment assignment per aspect

---

## ⚠️ Common Pitfalls

* Missing or duplicated `review_id`
* Invalid aspect names
* Mismatch between `aspects` and `aspect_sentiments`
* Invalid sentiment labels

---

## 💡 Design Considerations

* Modular architecture for scalability
* Domain-agnostic modeling approach
* Robust handling of noisy Arabic text
* Strict compliance with evaluation schema

---

## 👥 Team Contributions

* Data preprocessing and feature engineering
* Model design and training
* Inference pipeline and submission generation

---

## 🏁 Conclusion

This project delivers a **scalable and production-oriented ABSA system** capable of extracting meaningful insights from complex, real-world Arabic reviews. The solution balances model performance, robustness, and strict adherence to evaluation requirements.

---
