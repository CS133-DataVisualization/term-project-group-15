# 🎮 Predicting Video Game Pricing

Group 15: Seven (Stephani) Soriano, Satyansh Rai, Karla Nguyen, Jay Barrios Abarquez

How can we use predictive models to estimate a game’s price or classify it into pricing categories?

## 📌 Abstract
This project takes video prices using Steam dataset information to look at important pricing factors and build Machine Learning (ML) models for both regression and classification problems. After utilizing common preprocessing tools, we removed many leakage-prone price variables and utilized features such as release year, recommendations, Metacritic information, and genres to truly understand the pricing of games and how they are determined. Exploratory analysis illustrated that most games are lower priced, with relatively few premium titles, creating both skewness and class imbalance towards lower budget, lower priced games. We compared Linear Regression, XGBoost, and Random Forest models. In the end, Random Forest performed the best out of our analysis for price prediction, achieving a test RMSE of 7.61 and R² of 0.290, while also producing the strongest overall classification balance with 83.7% accuracy and an F1-score of 0.276. These results show that, while working well enough on “budget” games, the skewness makes it difficult to predict “premium” level games. In essence, it concludes that video game pricing is influenced by complex relationships, not something that can be described linearly, needing more detailed features.

## 📂 Project Setup

### 1️⃣ Install Dependencies
* Ensure all required Python libraries are installe:
  * `pip install pandas numpy matplotlib scikit-learn xgboost`
 
### 2️⃣ Place the dataset files in the correct directory:
* `Project Assignments/models/ml_splits/`

### 3️⃣ Ensure the dataset files are present
* X_train.csv
* X_test.csv
* y_reg_train.csv
* y_reg_test.csv
* y_cls_train.csv
* y_cls_test.csv

### 4️⃣ Run the Python scripts
* `python lin_reg.py`
* `python xgboost.py`
* `python random_forrest_regression.py`
