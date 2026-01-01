# House-Prediction

A machine learning project to predict house prices using Linear Regression. This project demonstrates the complete workflow of data preprocessing, model training, and evaluation.

## Project Overview

This project uses a Linear Regression model to estimate house prices based on various property features such as area, number of bedrooms, bathrooms, and amenities. The model is trained on historical housing data and evaluated on a test set to determine its predictive accuracy.

## Dataset

The project uses the `Housing.csv` file which contains information about various houses and their prices. The dataset includes both numerical features (area, price, etc.) and categorical features (yes/no amenities).

### Features in the Dataset
- **Numerical Features**: Area, bedrooms, bathrooms, stories, price
- **Binary Features**: 
  - mainroad
  - guestroom
  - basement
  - hotwaterheating
  - airconditioning
  - prefarea

## Process: House Price Estimation

### 1. **Data Loading**
   - Load the housing data from `Housing.csv` using pandas
   - The dataset is read into a DataFrame for processing

### 2. **Data Preprocessing**
   
   **Binary Column Conversion**:
   - Convert binary categorical features ("yes"/"no") to numerical values (1/0)
   - Columns converted: mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea
   - Example: "yes" → 1, "no" → 0
   
   **Categorical Encoding**:
   - Use one-hot encoding (via `pd.get_dummies()`) to convert any remaining categorical variables
   - `drop_first=True` removes redundant columns to avoid multicollinearity

### 3. **Feature and Target Separation**
   - **Features (X)**: All columns except "price"
   - **Target (y)**: The "price" column (what we want to predict)

### 4. **Train-Test Split**
   - Split the data into training (80%) and testing (20%) sets
   - `test_size=0.2`: 20% of data reserved for testing
   - `random_state=42`: Ensures reproducibility across runs
   - Training set is used to fit the model, test set evaluates its generalization

### 5. **Model Training**
   - **Algorithm**: Linear Regression
   - Fit the LinearRegression model using training data (X_train, y_train)
   - The model learns the linear relationship between features and price

### 6. **Predictions**
   - Use the trained model to make predictions on the test set
   - Generate predicted house prices for the test data

### 7. **Model Evaluation**
   - **Metric**: R² Score (Coefficient of Determination)
   - Measures how well the model explains the variance in house prices
   - R² ranges from 0 to 1, where 1 indicates perfect predictions
   - Higher R² score indicates better model performance

### 8. **Output**
   - Display first 5 predicted house prices
   - Display the model's R² accuracy score

## Requirements

- pandas
- scikit-learn

## How to Run

1. Ensure `Housing.csv` is in the same directory as `house_prediction.py`
2. Install required packages:
   ```bash
   pip install pandas scikit-learn
   ```
3. Run the script:
   ```bash
   python house_prediction.py
   ```

## Output

The script will display:
- The first 5 predicted house prices from the test set
- The R² score indicating model accuracy

## Model Interpretation

The Linear Regression model creates an equation of the form:
```
price = β₀ + β₁×feature₁ + β₂×feature₂ + ... + βₙ×featureₙ
```

Where:
- β₀ is the intercept
- β₁, β₂, ..., βₙ are the coefficients for each feature
- These coefficients show the relationship between each feature and the predicted price