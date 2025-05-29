# 📱 Mobile Price Prediction using Machine Learning

This project aims to predict the **actual price** of a mobile phone based on its technical specifications using regression algorithms. It explores the relationship between features like RAM, battery, and screen resolution with the phone price and builds models to forecast prices accurately.

---

## 🎯 Objective

To build a regression model that predicts the mobile phone price (continuous value) based on multiple features such as RAM, battery power, screen size, etc.

---

## 📦 Dataset Details

- **File used**: `Mobile-Price-Prediction-cleaned_data.csv`
- **Loaded using**: `pd.read_csv()`
- **Target variable**: `price` (continuous)

### 📊 Features Overview

| Feature          | Description                             |
|------------------|-----------------------------------------|
| battery_power     | Battery capacity in mAh                |
| blue              | Bluetooth availability (1 = Yes, 0 = No) |
| clock_speed       | Processor speed in GHz                 |
| dual_sim          | Dual SIM support                       |
| fc                | Front camera megapixels                |
| four_g            | 4G availability                        |
| int_memory        | Internal memory in GB                  |
| m_dep             | Mobile depth in cm                     |
| mobile_wt         | Mobile weight in grams                 |
| n_cores           | Number of CPU cores                    |
| pc                | Primary camera megapixels              |
| px_height         | Pixel height of the display            |
| px_width          | Pixel width of the display             |
| ram               | RAM in MB                              |
| sc_h              | Screen height in cm                    |
| sc_w              | Screen width in cm                     |
| talk_time         | Talk time in hours                     |
| three_g           | 3G availability                        |
| touch_screen      | Touchscreen availability               |
| wifi              | WiFi support                           |
| **price**         | 📌 Target — mobile price in units       |

---

## ⚙️ Project Workflow and Code Explanation

### 📌 Step 1: Import Libraries
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

📌 Step 2: Load Dataset
df = pd.read_csv('C:/10 aiml projects/Mobile-Price-Prediction-cleaned_data.csv')


📌 Step 3: Data Exploration
Checked the head and tail of the data
Verified there are no missing values:

df.isnull().sum()

📌 Step 4: Exploratory Data Analysis (EDA)
Correlation Matrix using df.corr() and sns.heatmap()

Pairwise feature relations:

Example: RAM shows a strong positive correlation with price_range

Outlier detection:

sns.boxplot(x='price_range', y='ram', data=df)

📌 Step 5: Data Preprocessing
Separated features (X) and target (y) 

X = df.drop(['price_range'], axis=1)
y = df['price_range']

sns.boxplot(x='price_range', y='ram', data=df)

Feature Scaling:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

📌 Step 6: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

📌 Step 7: Model Training and Evaluation


🧠 Model Training and Evaluation

✅ Linear Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

✅ Decision Tree

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

✅ Random Forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

📏 Evaluation
All models are evaluated using:

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
| Model               | Approx Accuracy |
| ------------------- | --------------- |
| Linear Regression | \~43.32%           |
| Decision Tree       | \~57.62%           |
| Random Forest       | **\~81.19%**       |

📌 Random Forest performed best due to its ensemble structure and handling of complex feature interactions.

# 🚀 Future Advancements

🔧 Improvements

Hyperparameter tuning with GridSearchCV

Add feature selection to reduce noise

Try ensemble methods like ExtraTreesClassifier

# 🌍 Deployment Plan

Build an interactive Streamlit or Flask web app

UI: Input fields for specifications, output: predicted price range

Host the app using:

 Streamlit Cloud

 Render or Heroku

 AWS EC2 or HuggingFace Spaces

# 💻 Run This Project Locally

# Clone the repo:


git clone https://github.com/sucharita-g/mobile-price-prediction.git
cd mobile-price-prediction

# Install dependencies:

pip install -r requirements.txt

# Launch Jupyter Notebook:

jupyter notebook "Mobile Price Prediction.ipynb"


# 📬 Author

**sucharita G**

🔗 https://www.linkedin.com/in/sucharita-g-105517257/

📧 sucharitagopi@gmail.com

