# 🧠 MindWatch  
### AI-Powered Mental Health Platform

MindWatch is a comprehensive web application that uses Artificial Intelligence (DistilBERT) to detect early signs of depression from user text.

The platform provides real-time analysis, emotional tracking, private journaling, self-help resources, and crisis support — while maintaining strict privacy and ethical standards.

---

## 🚀 Features

### 🤖 AI Depression Detection
- Real-time text analysis using fine-tuned DistilBERT
- Emotion classification with confidence score
- Crisis keyword detection system

### 📊 Emotional Timeline
- Visual mood tracking over time
- Historical emotional trend analysis
- Interactive charts powered by Chart.js

### 📓 Private Journal
- Secure journaling space
- Automatic emotion tagging
- Optional anonymous mode

### 🧘 Self-Help Resources
- Guided breathing exercises
- Wellness activities
- Crisis helpline directory

### 🔒 Privacy First
- Anonymous mode support
- Secure data handling
- Ethical AI usage

### 🎨 Modern UI
- Glassmorphism design
- Responsive layout
- Dark mode aesthetics

---

## 🛠️ Technology Stack

| Layer | Technology |
|--------|------------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python, Flask |
| AI/ML | PyTorch, Hugging Face Transformers, DistilBERT |
| Database | SQLite (Dev) / PostgreSQL (Prod) |
| Visualization | Chart.js |

---

## 📋 Prerequisites

- Python 3.9+
- pip

---

## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/mindwatch.git
cd mindwatch
```
### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Environment Variables (Optional)

```bash
cp .env.example .env
```

---

## 🧠 Model Setup

Before running the application, train the model:

```bash
python ml/train_model.py
```
---

## 🏃 Running the Application

Start the Flask server:

```bash
python app.py
```

Then open in your browser:

```
http://localhost:5000
```

---

## 🧪 Testing

| Scenario | Example Input |
|----------|--------------|
| Positive Mood | "I feel great today!" |
| Negative Mood | "I feel hopeless." |
| Crisis Detection | "I want to end it all." |
| Anonymous Mode | Login via Anonymous Mode |

---



## 📄 License

This project is for educational purposes only.  
Ensure compliance with API and dataset licensing terms before public deployment.
