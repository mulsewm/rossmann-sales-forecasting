# Quick Start Guide - Rossmann Sales Forecasting

##  Project Setup Complete!

##  Project Structure

```
rossmann-sales-forecasting/
├── data/
│   ├── raw/                 #  Kaggle data here
│   └── processed/           # Processed data will be saved here
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_modeling.ipynb
│   ├── 04_deep_learning_lstm.ipynb
│   └── 05_model_evaluation.ipynb
├── src/                     # Source code modules
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and prediction
│   └── visualization/      # Plotting functions
├── api/                     # FastAPI application
├── models/                  # Saved models
├── tests/                   # Unit tests
├── scripts/                 # Utility scripts
├── logs/                    # Log files
└── docs/                    # Documentation
```

##  Getting Started

### Step 1: Set Up Environment

```bash
# Navigate to project directory
cd rossmann-sales-forecasting

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### Step 2: Configure Kaggle API

```bash
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

# Set permissions (macOS/Linux)
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Data

```bash
# Download Rossmann data from Kaggle
python scripts/download_data.py --validate

# Or manually download from:
# https://www.kaggle.com/c/rossmann-store-sales/data
# And place files in data/raw/
```

### Step 4: Explore the Data

```bash
# Start Jupyter Lab
jupyter lab

# Open: notebooks/01_data_exploration.ipynb
# Run all cells to perform EDA
```

### Step 5: Train Models

```bash
# Option 1: Use the training pipeline (recommended)
python scripts/train_pipeline.py --model all

# Option 2: Quick test with reduced parameters
python scripts/train_pipeline.py --model xgboost --quick-mode

# Option 3: Train specific model
python scripts/train_pipeline.py --model lightgbm --save-predictions

# Options:
#   --model: rf, xgboost, lightgbm, or all
#   --quick-mode: Fast training for testing
#   --save-predictions: Save predictions to CSV
#   --feature-importance: Generate feature importance files
```

### Step 6: Launch API

```bash
# Navigate to API directory
cd api

# Start the API server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Access documentation at:
# http://localhost:8000/docs
```

##  Workflow

### Data Science Workflow:

1. **Explore** → `notebooks/01_data_exploration.ipynb`
   - Understand the data
   - Identify patterns
   - Find data quality issues

2. **Engineer Features** → `notebooks/02_feature_engineering.ipynb`
   - Create time-based features
   - Create lag and rolling features
   - Handle missing values

3. **Train ML Models** → `notebooks/03_ml_modeling.ipynb`
   - Random Forest
   - XGBoost
   - LightGBM

4. **Train Deep Learning** → `notebooks/04_deep_learning_lstm.ipynb`
   - LSTM time series model
   - Compare with ML models

5. **Evaluate** → `notebooks/05_model_evaluation.ipynb`
   - Compare all models
   - Analyze errors
   - Select best model

### Production Workflow:

```bash
# 1. Train model
python scripts/train_pipeline.py --model all

# 2. Test API locally
cd api
uvicorn app:app --reload

# 3. Make predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Store": 1,
      "DayOfWeek": 5,
      "Open": 1,
      "Promo": 1,
      "StateHoliday": "0",
      "SchoolHoliday": 0
    }
  }'
```

##  Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_models.py -v
```

##  Key Files

| File | Description |
|------|-------------|
| `README.md` | Main project documentation |
| `QUICKSTART.md` | This file - quick start guide |
| `docs/project_plan.md` | Detailed project plan and timeline |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation configuration |
| `.gitignore` | Git ignore rules |

## 🔧 Common Commands

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Run code quality checks
black src/  # Format code
flake8 src/  # Lint code
pylint src/  # Advanced linting

# Start Jupyter Lab
jupyter lab

# Start API server
cd api && uvicorn app:app --reload

# Run training pipeline
python scripts/train_pipeline.py --model all
```

##  API Usage Examples

### Python Example:

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Request data
data = {
    "features": {
        "Store": 1,
        "DayOfWeek": 5,
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
        "StoreType": "c",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
        "Promo2": 0
    }
}

# Make request
response = requests.post(url, json=data)
print(response.json())
```

### cURL Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Store": 1,
      "DayOfWeek": 5,
      "Open": 1,
      "Promo": 1
    }
  }'
```

##  Troubleshooting

### Issue: Kaggle API not working
```bash
# Solution: Check credentials
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Import errors
```bash
# Solution: Install in development mode
pip install -e .
```

### Issue: Notebook can't find modules
```python
# Solution: Add to notebook
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
```

### Issue: API won't start
```bash
# Solution: Check if port is in use
lsof -i :8000  # macOS/Linux
# Kill the process or use a different port
uvicorn app:app --reload --port 8001
```

##  Next Steps

1. **Explore the data** using notebooks
2. **Train your first model** with the pipeline
3. **Deploy the API** for predictions
4. **Customize features** based on insights
5. **Tune hyperparameters** for better performance
6. **Monitor model** performance over time

##  Tips

- Start with `--quick-mode` for faster iterations
- Use a subset of data during development
- Check logs in `logs/` directory
- Save models incrementally during training
- Document your findings in notebooks
- Use git for version control

##  Need Help?

- Check `README.md` for detailed documentation
- Review `docs/project_plan.md` for methodology
- Look at docstrings in source code
- Run `pytest` to see test examples
- Check API docs at `http://localhost:8000/docs`

##  Project Goals

- **RMSPE < 10%** on test set
- **Production-ready API** with <100ms response time
- **Comprehensive documentation** for reproducibility
- **Clean, maintainable code** with tests

---

**Good luck with your project! **

For questions or issues, refer to the documentation in `README.md` and `docs/project_plan.md`.

