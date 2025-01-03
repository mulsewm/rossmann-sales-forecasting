# Rossmann Pharmaceuticals - Interim Report (Task 1: Exploratory Data Analysis)

## 1. Overview
The goal of this project is to forecast sales for Rossmann Pharmaceuticals' stores across different cities six weeks ahead. This report focuses on Exploratory Data Analysis (EDA) to identify patterns, trends, and potential features for predictive modeling.

---

## 2. Data Summary
- **Datasets**: 
  - `train.csv` – Sales data at the store level.
  - `store.csv` – Store-specific information (e.g., competition, assortment, etc.).
- **Merging**: The `train` and `store` datasets were merged using the `Store` column.

- **Preprocessing**: 
  - Handled missing values by filling `CompetitionDistance` with the median.
  - Extracted date features: `Year`, `Month`, `DayOfWeek`, and `WeekOfYear`.

---

## 3. Key Insights and Visualizations
### 3.1. Sales Distribution by Promo
- **Insight**: Stores running promotions (`Promo=1`) exhibit higher sales compared to non-promotional days.
- **Visualization**:  
  ![](../visualizations/promo_vs_sales.png)

---

### 3.2. Impact of Holidays on Sales
- **Insight**: Sales are higher during public holidays (`StateHoliday=a`) but significantly lower during school holidays.
- **Visualization**:  
  ![](../visualizations/holiday_vs_sales.png)

---

### 3.3. Correlation Between Customers and Sales
- **Insight**: There is a strong positive correlation between the number of customers and daily sales. Stores with high customer traffic tend to have higher sales.
- **Visualization**:  
  ![](../visualizations/customers_vs_sales.png)

---

### 3.4. Additional Observations:
- **Day of the Week**: Sales are highest at the beginning of the week (Monday) and decline gradually towards Sunday.  
- **Assortment Types**: Stores with extended assortments (`Assortment=c`) tend to have higher sales compared to stores with basic assortments.  
- **Competition Distance**: Stores with closer competitors tend to have slightly lower sales.  

---

## 4. Next Steps (Task 2: Modeling)
- Train machine learning models to forecast sales.
- Engineer additional features such as:
  - Number of days until the next holiday.
  - Rolling average of sales over the past 7 days.
- Implement hyperparameter tuning to optimize model performance.
