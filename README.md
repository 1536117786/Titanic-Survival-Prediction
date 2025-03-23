# 🚢 Titanic Survival Prediction – Classification with Machine Learning
This notebook focuses on building and evaluating multiple machine learning models to predict whether a passenger survived the Titanic disaster. It follows a structured pipeline from data loading to final evaluation and model comparison.

## 📌 Project Objective
The goal is to accurately predict the survival status of Titanic passengers (Survived) based on features such as age, sex, class, and fare using classification algorithms. This is a classic binary classification task (0 = did not survive, 1 = survived).

## 🧰 Tools & Libraries Used
Python
Pandas & NumPy – Data loading and transformation
Seaborn & Matplotlib – Data visualization
Scikit-learn – ML models, preprocessing, evaluation
## 🧪 Steps in the Notebook
1. Data Loading & Exploration
Data is loaded using pandas.
A quick .info() and .describe() is used to understand datatypes, null values, and statistical distributions.
2. Exploratory Data Analysis (EDA)
Checked for missing values and imbalanced classes.
Created visualizations:
Survival rate by gender and Pclass
Distribution of Age, Fare, and Embarked
Key Observations:
Females had a much higher survival rate than males.
Passengers in 1st class had better chances of survival.
3. Data Preprocessing
Imputation: Filled missing Age with mean or median, Embarked with the mode.
Encoding:
Converted categorical variables (Sex, Embarked) using Label Encoding or OneHotEncoding.
Scaling:
Standardized numerical features (Age, Fare) using StandardScaler.
4. Model Building
Trained and evaluated several classifiers:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Support Vector Machine (SVM)
Naive Bayes
Each model was trained using a train-test split, and performance was evaluated.

## 📈 Model Evaluation Metrics
Each model was evaluated using:

Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
Model	Accuracy Score
Logistic Regression	✅ 81%
K-Nearest Neighbors	✅ 78%
Decision Tree	✅ 76%
Random Forest	✅ 83%
SVM (RBF kernel)	✅ 82%
Naive Bayes	✅ 78%
## 🔍 Best Model: Random Forest achieved the highest accuracy and performed well across other metrics, especially in handling feature interactions.

## ✅ Key Results & Interpretation
Gender and class were the most influential features.
Random Forest and SVM outperformed other models in prediction accuracy.
Feature scaling helped SVM and KNN perform better.
Naive Bayes was fast but less accurate due to feature independence assumptions.
## 📊 Future Improvements
Apply GridSearchCV or RandomizedSearchCV for hyperparameter tuning.
Implement Cross-validation for more robust evaluation.
Explore Ensemble models like XGBoost or LightGBM.
Try feature engineering (e.g., combining SibSp and Parch into FamilySize).
## 📁 Project Structure
Copy
Edit
.
├── Titanic_Classification.ipynb
└── README.md
## 🤝 Contributing
Feel free to fork, improve, and experiment with the notebook. Pull requests are welcome!

