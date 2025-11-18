#  Rossmann Sales Forecasting with Machine Learning & Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)
![License](https://img.shields.io/badge/License-MIT-green)

##  Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Project Timeline](#project-timeline)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

##  Project Overview

This project focuses on forecasting daily sales for **Rossmann Pharmaceuticals** stores 6 weeks ahead using Machine Learning and Deep Learning techniques. Rossmann operates over 3,000 drug stores across 7 European countries, and accurate sales forecasting is crucial for inventory management, staffing, and strategic planning.

##  Business Objective

The finance team at Rossmann Pharmaceuticals wants to forecast sales in all their stores across different cities six weeks ahead of time. Store managers currently predict their daily sales based on their experience, but the accuracy varies significantly. This project aims to:

- Build robust ML models (Random Forest, XGBoost, LightGBM) for baseline predictions
- Implement Deep Learning models (LSTM) for capturing temporal patterns
- Deploy a production-ready API for real-time predictions
- Provide actionable insights through comprehensive EDA

##  Dataset Information

**Source:** [Rossmann Store Sales - Kaggle Competition](https://www.kaggle.com/c/rossmann-store-sales)

**Files:**
- `train.csv`: Historical sales data (2013-01-01 to 2015-07-31)
- `test.csv`: Store information without sales (for predictions)
- `store.csv`: Supplemental information about the stores

**Key Features:**
- **Store**: Unique ID for each store
- **Sales**: Turnover for any given day (target variable)
- **Customers**: Number of customers on a given day
- **Open**: Whether store was open (0 = closed, 1 = open)
- **Promo**: Whether store was running a promo
- **StateHoliday**: State holiday indicator
- **SchoolHoliday**: Affects store operations
- **StoreType, Assortment**: Store characteristics
- **CompetitionDistance**: Distance to nearest competitor
- **Promo2**: Continuing and consecutive promotion

**Dataset Size:** 
- Training: ~1M records
- Test: ~41K records

##  Project Structure

```
rossmann-sales-forecasting/
│
├── data/
│   ├── raw/                    # Original datasets from Kaggle
│   └── processed/              # Cleaned and feature-engineered data
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_modeling.ipynb
│   ├── 04_deep_learning_lstm.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                        # Source code modules
│   ├── __init__.py            # Logging configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py       # Data loading from Kaggle
│   │   └── preprocessing.py   # Data cleaning utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py     # ML model training
│   │   ├── predict_model.py   # Inference pipeline
│   │   └── lstm_model.py      # Deep learning models
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py       # Plotting functions
│
├── models/                     # Serialized trained models
│   └── .gitkeep
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── app.py                 # API endpoints
│   └── requirements.txt       # API-specific dependencies
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_models.py
│
├── logs/                       # Application logs
│   └── .gitkeep
│
├── scripts/                    # Utility scripts
│   ├── download_data.py       # Download from Kaggle
│   └── train_pipeline.py      # Full training pipeline
│
├── docs/                       # Documentation
│   └── project_plan.md        # Detailed project plan
│
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── LICENSE                     # MIT License
└── README.md                   # This file
```

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Kaggle API credentials (for data download)

### Step 1: Clone the Repository
```bash
git clone https://github.com/mulsewm/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n rossmann python=3.10
conda activate rossmann
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Package in Development Mode
```bash
pip install -e .
```

### Step 5: Setup Kaggle API
1. Create a Kaggle account and generate API token
2. Download `kaggle.json` from your Kaggle account settings
3. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<Username>\.kaggle\kaggle.json` (Windows)
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Step 6: Download Data
```bash
python scripts/download_data.py
```

##  Usage

### 1. Exploratory Data Analysis
```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

### 2. Feature Engineering
```bash
jupyter lab notebooks/02_feature_engineering.ipynb
```

### 3. Train Machine Learning Models
```bash
# Run the full training pipeline
python scripts/train_pipeline.py

# Or train specific models
python -m src.models.train_model --model xgboost
```

### 4. Train Deep Learning Models
```bash
python -m src.models.lstm_model --epochs 50 --batch_size 64
```

### 5. Make Predictions
```bash
python -m src.models.predict_model --model_path models/best_model.pkl --data_path data/test.csv
```

### 6. Launch API Server
```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then visit: `http://localhost:8000/docs` for interactive API documentation

### 7. Run Tests
```bash
pytest tests/ -v --cov=src
```

##  Technologies Used

### Data Processing & Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations

### Machine Learning
- **scikit-learn**: Classical ML algorithms, preprocessing, metrics
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting

### Deep Learning
- **TensorFlow/Keras**: Deep learning framework (LSTM models)
- **PyTorch**: Alternative DL framework (optional)

### API Development
- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation
- **uvicorn**: ASGI server

### MLOps & Utilities
- **Kaggle**: Data download
- **python-dotenv**: Environment management
- **loguru**: Advanced logging
- **pytest**: Testing framework

##  Project Timeline

| Phase | Tasks | Duration | Status |
|-------|-------|----------|--------|
| **Week 1-2** | Data Collection, EDA, Cleaning | 2 weeks | 🟡 In Progress |
| **Week 3** | Feature Engineering | 1 week | ⚪ Pending |
| **Week 4** | ML Model Development (RF, XGBoost) | 1 week | ⚪ Pending |
| **Week 5** | Deep Learning (LSTM) | 1 week | ⚪ Pending |
| **Week 6** | Model Evaluation & Selection | 1 week | ⚪ Pending |
| **Week 7** | API Development & Deployment | 1 week | ⚪ Pending |
| **Week 8** | Documentation & Presentation | 1 week | ⚪ Pending |

##  Results

*This section will be updated with model performance metrics after training*

### Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD |

### Key Insights
- TBD: Add insights from EDA
- TBD: Feature importance analysis
- TBD: Temporal patterns discovered

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Mulusew Meselu Tesfaye** - *Initial work* - [YourGitHub](https://github.com/mulsewm)

##  Acknowledgments

- Kaggle for providing the Rossmann Store Sales dataset
- Rossmann Pharmaceuticals for the business case
- Open source community for the amazing tools and libraries

##  Contact

For questions or feedback, please reach out:
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/muliemes)
- GitHub: [@mulsewm](https://github.com/mulsewm)

---

⭐ Star this repository if you find it helpful!

