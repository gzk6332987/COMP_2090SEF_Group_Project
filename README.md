# Text Emotion Classification App (PyTorch)

## 📌 Project Overview
This project is developed for **COMP2090SEF/8090SEF Group Project** at HKMU. 
It is a Python-based application that performs **Real-time Emotion Classification** on text input. By leveraging Deep Learning (PyTorch) and Natural Language Processing (NLP), the app identifies emotions (e.g., Joy, Sadness, Anger) from user-provided sentences.

---

## 🛠️ Technical Stack
- **Language:** Recommend Python 3.10+ (We can't make sure this program can run in low python version)
- **Deep Learning Framework:** PyTorch
- **Data Processing:** Hashing Trick (MD5-based) for OOV management
- **Version Control:** Git & GitHub

---

## 🏗️ System Architecture & OOP Concepts
Following the course requirements, this project is built using strictly **Object-Oriented Programming (OOP)** principles across multiple modules:

1. **Encapsulation (`data_processor.py`):** 
   - Uses the `TextPreprocessor` class to hide complex text cleaning and hashing logic.
2. **Inheritance (`model.py`):** 
   - Our `EmotionClassifier` class inherits from `torch.nn.Module`, the base class for all neural network modules in PyTorch.
3. **Abstraction (`app_engine.py`):** 
   - Provides a simplified interface for the end-user to interact with the model without knowing the underlying tensor operations.
4. **Modularity:** 
   - The project is split into at least 3 distinct modules to ensure high maintainability.

---

## 🚀 Installation & Usage Guide

### 1. Prerequisites
Ensure you have [Conda](https://docs.anaconda.com) or [Python 3.10+](https://www.python.org) installed.

### 2. Setup Environment
Clone the repository and install dependencies:
```bash
# Clone the repo
git clone https://github.com[Your-Github-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
