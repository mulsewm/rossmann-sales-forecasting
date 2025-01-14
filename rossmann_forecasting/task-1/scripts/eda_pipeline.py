import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Ensure the reports directory exists
if not os.path.exists('../reports'):
    os.makedirs('../reports')

# Set up logging
logging.basicConfig(
    filename='../reports/eda.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# 1. Load Data
def load_data():
    try:
        train = pd.read_csv('../data/train.csv')
        store = pd.read_csv('../data/store.csv')
        logging.info("Train and Store data loaded successfully")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

    # Merge train and store data
    data = pd.merge(train, store, on='Store', how='left')
    logging.info("Train and Store datasets merged")
    
    return data


# 2. Data Preprocessing and Feature Engineering
def preprocess_data(data):
    try:
        # Handle missing values
        data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
        data.fillna(0, inplace=True)
        logging.info("Missing values handled successfully")

        # Convert Date column to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Feature Engineering - Extract date features
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week

        logging.info("Feature engineering completed")
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
    
    return data


# 3. Visualizations

# 3.1. Sales Distribution by Promo
def plot_sales_promo(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Promo', y='Sales', data=data)
    plt.title('Sales Distribution by Promo')
    plt.savefig('../visualizations/promo_vs_sales.png')
    plt.show()
    logging.info("Promo vs Sales visualization saved")


# 3.2. Holiday Impact on Sales
def plot_sales_holiday(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StateHoliday', y='Sales', data=data)
    plt.title('Holiday Impact on Sales')
    plt.savefig('../visualizations/holiday_vs_sales.png')
    plt.show()
    logging.info("Holiday vs Sales visualization saved")


# 3.3. Correlation between Customers and Sales
def plot_correlation(data):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Customers', y='Sales', data=data)
    plt.title('Correlation between Customers and Sales')
    plt.savefig('../visualizations/customers_vs_sales.png')
    plt.show()
    logging.info("Customers vs Sales correlation plot saved")


# 3.4. Sales by Day of the Week
def plot_sales_by_day(data):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='DayOfWeek', y='Sales', data=data)
    plt.title('Sales by Day of the Week')
    plt.savefig('../visualizations/sales_by_day.png')
    plt.show()
    logging.info("Sales by Day of the Week plot saved")


# 3.5. Assortment Type and Sales
def plot_assortment_sales(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Assortment', y='Sales', data=data)
    plt.title('Sales by Assortment Type')
    plt.savefig('../visualizations/assortment_vs_sales.png')
    plt.show()
    logging.info("Assortment vs Sales visualization saved")


# 3.6. Competition Distance vs Sales
def plot_competition_distance(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=data)
    plt.title('Competition Distance vs Sales')
    plt.savefig('../visualizations/competition_vs_sales.png')
    plt.show()
    logging.info("Competition Distance vs Sales visualization saved")


# 4. Main Execution Pipeline
if __name__ == "__main__":
    # Load data
    data = load_data()

    if data is not None:
        # Preprocess data
        data = preprocess_data(data)

        # Ensure visualizations directory exists
        if not os.path.exists('../visualizations'):
            os.makedirs('../visualizations')

        # Generate visualizations
        plot_sales_promo(data)
        plot_sales_holiday(data)
        plot_correlation(data)
        plot_sales_by_day(data)
        plot_assortment_sales(data)
        plot_competition_distance(data)

        logging.info("EDA pipeline completed successfully")
    else:
        logging.error("EDA pipeline terminated due to data loading failure")
