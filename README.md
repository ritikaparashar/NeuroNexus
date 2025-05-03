# 🚢 Titanic Survival Prediction
This project uses the Titanic dataset to build a machine learning model that predicts whether a passenger survived the Titanic disaster.It uses the famous Titanic dataset and is ideal for beginners in data science
. It leverages a **Random Forest Classifier** and explores key data insights through visualizations.

---

## 📁 Dataset

The dataset (`tested.csv`) contains details about passengers, including:

- Passenger class (Pclass)
- Sex
- Age
- Fare
- Embarked location
- Survival status (0 = No, 1 = Yes)

---

## 🔧 Technologies Used

- Python
- Pandas & NumPy
- Seaborn & Matplotlib
- Scikit-learn

---

## 📊 Data Preprocessing

- Missing values in `Age` and `Fare` filled with median.
- Categorical columns `Sex` and `Embarked` encoded using `LabelEncoder`.
- Irrelevant columns like `Cabin`, `Name`, `Ticket`, `PassengerId` were dropped.

---

## ⚙️ Model Training

A **Random Forest Classifier** was used with 100 estimators and trained using an 80-20 train-test split.

---

## ✅ Evaluation Metrics

- **Accuracy**
- **Classification Report**
- **Confusion Matrix**

---

## 📌 Feature Importance

Understanding which features contributed most to the prediction:

![Feature Importance](feature_importance.png)

---

## 📂 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `tested.csv` is in the project directory.

4. Run the script:
   ```bash
   python titanic_survival_prediction.py
   ```

---

## 🧠 Insights

- **Sex** and **Pclass** were the most influential features.
- Survival chances were higher for:
  - Females
  - First-class passengers
  - Younger passengers

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Add ensemble comparisons (e.g., XGBoost, SVM)
- Build a web app using Streamlit or Flask

---

## 🧾 License

This project is licensed under the MIT License. Feel free to use and modify.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.
