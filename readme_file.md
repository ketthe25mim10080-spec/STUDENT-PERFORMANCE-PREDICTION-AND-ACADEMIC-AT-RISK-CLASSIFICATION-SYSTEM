# Student Academic Performance & Risk Prediction System

A machine learning pipeline designed to predict student performance outcomes and flag individuals at risk of academic failure based on daily habits, attendance, and past performance.

## Requirements & Environment Setup

### Prerequisites
* Python 3.8 or higher
* `pip` package manager

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install numpy pandas scikit-learn joblib
   ```

---

## Execution Instructions

The project is fully executable via the command line (`CLI`).

### 1. Generate Synthetic Dataset
If no dataset is provided, generate a sample dataset:
```bash
python main.py --generate-data
```

### 2. Train the Machine Learning Model
Train the Random Forest classification model and save the artifact:
```bash
python main.py --train
```

### 3. Run Inference / Predictions
Run single predictions directly via CLI flags:
```bash
python main.py --predict --study 6.5 --attendance 82.0 --past_score 70.0 --extra 1 --sleep 7.5
```

---

## Project Structure

```
├── main.py              # Main executable CLI pipeline
├── student_data.csv     # Dataset generated or used for training
├── student_model.pkl    # Serialized model checkpoint and scaler
├── PROJECT_REPORT.md    # Detailed technical project report
└── README.md            # Setup and execution guide
```