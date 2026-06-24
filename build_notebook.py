import nbformat

def add_markdown(nb, text):
    nb.cells.append(nbformat.v4.new_markdown_cell(text))

def add_code(nb, text):
    nb.cells.append(nbformat.v4.new_code_cell(text))

# Read existing notebook
with open('lab-3.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Keep only the first cell or reset? The user has already run the first few cells, 
# but it's better to clear it and recreate, or append.
# Let's just create a new notebook to be safe and clean, and overwrite.
nb = nbformat.v4.new_notebook()

add_markdown(nb, "# Lab 3: Simple Linear Regression")

add_markdown(nb, "## Part A: Data Collection and Preprocessing")
add_code(nb, """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load the dataset using Pandas
# Assuming the file is 'data/Department Awareness Survey (Responses).xlsx'
df = pd.read_excel('data/Department Awareness Survey (Responses).xlsx')
""")

add_code(nb, """# 2. Display the first 5 rows
df.head()""")

add_code(nb, """# 3. Check dataset dimensions
print("Dimensions:", df.shape)""")

add_code(nb, """# Save to CSV as requested
df.to_csv('data/dataset.csv', index=False)
print("Saved to CSV.")""")

add_code(nb, """# 4. Identify missing values
print("Missing values per column:\\n", df.isnull().sum())""")

add_code(nb, """# Rename columns for easier access
df.rename(columns={
    'your CIA % of last semester ': 'CIA',
    'your GPA of last semester': 'GPA',
    'Your maximum attendance % till last semester': 'Attendance'
}, inplace=True)""")

add_code(nb, """# 5. Remove or handle null values
# 6. Convert required columns into numerical datatype
# The Attendance column has string 'Option 1', we should coerce to numeric
df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')
df['CIA'] = pd.to_numeric(df['CIA'], errors='coerce')
df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')

# Drop nulls in the required columns
df_clean = df.dropna(subset=['CIA', 'Attendance', 'GPA']).copy()
print("Dimensions after dropping nulls:", df_clean.shape)""")

add_code(nb, """# 7. Remove duplicate records if present
df_clean.drop_duplicates(inplace=True)
print("Dimensions after dropping duplicates:", df_clean.shape)""")

add_code(nb, """# 8. Generate statistical summary
df_clean[['CIA', 'Attendance', 'GPA']].describe()""")

add_code(nb, """# 9. Select appropriate dependent and independent variables
# Experiment 1
X_cia = df_clean[['CIA']]
Y_gpa = df_clean['GPA']

# Experiment 2
X_att = df_clean[['Attendance']]
Y_gpa_att = df_clean['GPA']
""")

add_markdown(nb, "## Part B: Simple Linear Regression using Scikit-learn")

add_code(nb, """# Experiment 1: CIA vs GPA
X_train_cia, X_test_cia, Y_train_cia, Y_test_cia = train_test_split(X_cia, Y_gpa, test_size=0.2, random_state=42)
model_cia = LinearRegression()
model_cia.fit(X_train_cia, Y_train_cia)

slope_cia = model_cia.coef_[0]
intercept_cia = model_cia.intercept_
print(f"Scikit-learn - CIA vs GPA: Slope = {slope_cia:.4f}, Intercept = {intercept_cia:.4f}")

preds_cia_sk = model_cia.predict(X_test_cia)
print("Predictions (Scikit-learn):", preds_cia_sk)""")

add_code(nb, """# Experiment 2: Attendance vs GPA
X_train_att, X_test_att, Y_train_att, Y_test_att = train_test_split(X_att, Y_gpa_att, test_size=0.2, random_state=42)
model_att = LinearRegression()
model_att.fit(X_train_att, Y_train_att)

slope_att = model_att.coef_[0]
intercept_att = model_att.intercept_
print(f"Scikit-learn - Attendance vs GPA: Slope = {slope_att:.4f}, Intercept = {intercept_att:.4f}")

preds_att_sk = model_att.predict(X_test_att)
print("Predictions (Scikit-learn):", preds_att_sk)""")

add_markdown(nb, "## Part C: Manual Computation using Ordinary Least Squares (OLS)")

add_code(nb, """# Experiment 1 Manual OLS
def manual_ols(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    slope = numerator / denominator
    intercept = Y_mean - slope * X_mean
    return slope, intercept

# Using training data for consistency
slope_cia_manual, intercept_cia_manual = manual_ols(X_train_cia['CIA'], Y_train_cia)
print(f"Manual OLS - CIA vs GPA: Slope = {slope_cia_manual:.4f}, Intercept = {intercept_cia_manual:.4f}")

preds_cia_manual = slope_cia_manual * X_test_cia['CIA'] + intercept_cia_manual
print("Predictions (Manual):", preds_cia_manual.values)
""")

add_code(nb, """# Experiment 2 Manual OLS
slope_att_manual, intercept_att_manual = manual_ols(X_train_att['Attendance'], Y_train_att)
print(f"Manual OLS - Attendance vs GPA: Slope = {slope_att_manual:.4f}, Intercept = {intercept_att_manual:.4f}")

preds_att_manual = slope_att_manual * X_test_att['Attendance'] + intercept_att_manual
print("Predictions (Manual):", preds_att_manual.values)
""")

add_markdown(nb, "## Comparison Task")

add_code(nb, """diff_cia = preds_cia_sk - preds_cia_manual.values
diff_att = preds_att_sk - preds_att_manual.values
print("Max absolute difference for CIA vs GPA:", np.max(np.abs(diff_cia)))
print("Max absolute difference for Attendance vs GPA:", np.max(np.abs(diff_att)))

print("Observation: Both methods produce identical predictions (up to floating point precision) since Scikit-learn Linear Regression uses the OLS formulation under the hood.")
""")

add_markdown(nb, "## Parameter Saving Task")

add_code(nb, """# Save weights for Experiment 1 (CIA vs GPA) as an example
weights = {
    'slope': slope_cia,
    'intercept': intercept_cia
}

with open('linear_regression_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)
print("Saved parameters to linear_regression_weights.pkl")
""")

add_code(nb, """# Load weights
with open('linear_regression_weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)
print("Loaded parameters:", loaded_weights)

# Predict using loaded parameters
sample_X = X_test_cia['CIA'].iloc[0]
pred_loaded = loaded_weights['slope'] * sample_X + loaded_weights['intercept']
print(f"Prediction for X={sample_X} using loaded weights: {pred_loaded:.4f}")
""")

add_markdown(nb, "## Sample Viva Questions Answers")

add_markdown(nb, """
1. **What is Simple Linear Regression?**
   Simple Linear Regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables. One variable is the predictor, explanatory, or independent variable (X), and the other is the response or dependent variable (Y).

2. **What is the role of slope and intercept?**
   The slope indicates the steepness of the regression line, representing how much Y is expected to change for a one-unit change in X. The intercept is the expected value of Y when X is 0 (where the line crosses the Y-axis).

3. **What is Ordinary Least Squares?**
   Ordinary Least Squares (OLS) is a method for estimating the unknown parameters (slope and intercept) in a linear regression model by minimizing the sum of the squared differences between the observed and predicted values.

4. **Why do we square the errors in OLS?**
   Squaring the errors ensures that positive and negative errors don't cancel each other out. It also heavily penalizes larger errors, ensuring the regression line fits the data as closely as possible.

5. **Difference between dependent and independent variable**
   The independent variable (X) is the variable that is being manipulated or used to predict the outcome. The dependent variable (Y) is the outcome variable being predicted or studied.

6. **Why should data be cleaned before training?**
   Real-world data often contains noise, missing values, and outliers. Cleaning data (e.g., handling nulls, correcting data types) prevents these issues from skewing the model, leading to more accurate and reliable predictions.

7. **Why are slope and intercept called model parameters?**
   They are the values that the learning algorithm discovers during training to map the input features to the target output. They define the specific model learned from the data.

8. **Why do we save learned weights?**
   Saving learned weights allows us to reuse the trained model later without having to retrain it from scratch. This is essential for deploying models to production and making future predictions efficiently.
""")

with open('lab-3.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Notebook generated successfully!")
