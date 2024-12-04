Results Using CMD :![image](https://github.com/user-attachments/assets/61314f1f-95c0-4fd0-961d-3c8151954c00)
### README File for Diabetes Prediction Model

---

## **Diabetes Prediction Model**

### **Overview**
This is a Python-based logistic regression model designed to predict the likelihood of diabetes in individuals. The project utilizes the Pima Indians Diabetes Database, which contains medical data such as glucose levels, BMI, and insulin levels, to predict diabetes outcomes. The model applies preprocessing to handle missing or invalid data and provides predictions with accuracy and metrics like sensitivity and specificity.

---

### **Features**
1. Trains a logistic regression model on the Pima Indians Diabetes Dataset.
2. Evaluates model performance using accuracy, sensitivity (true positive rate), and specificity (true negative rate).
3. Prompts users to input their own medical data for prediction.
4. Outputs predictions along with confidence in the model's performance.

---

### **Installation**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/<username>/diabetes-prediction
   ```
2. **Navigate to the project directory**:
   ```bash
   cd diabetes-prediction
   ```
3. **Install dependencies**:
   Ensure you have Python 3.x installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**
1. **Prepare your data**:
   - Download the Pima Indians Diabetes Dataset as a `.csv` file.
   - Ensure the file contains columns: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and `Outcome`.

2. **Run the script**:
   ```bash
   python CODE.py data.csv
   ```

3. **Interact with the model**:
   - After training the model, the script will prompt you to input medical data for a prediction. Follow the prompts to enter the following details:
     - Number of pregnancies
     - Glucose level
     - Blood pressure
     - Skin thickness
     - Insulin level
     - BMI
     - Diabetes Pedigree Function
     - Age

4. **Interpret the results**:
   - The script will output whether the individual is likely diabetic ("Positive") or not ("Negative") along with the model's accuracy.

---

### **Algorithm Used**

The model employs **Logistic Regression**, a supervised learning algorithm used for binary classification tasks. Logistic regression works by fitting a linear model to the data and using the logistic (sigmoid) function to output probabilities for binary outcomes (diabetic or non-diabetic in this case).

**Key Steps:**
1. **Data Preprocessing**:
   - Removes rows with invalid values (e.g., 0 for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, and `Age`).
2. **Feature Extraction**:
   - Splits the dataset into evidence (features) and labels (target outcomes).
3. **Training**:
   - Splits the data into training and test sets.
   - Trains the logistic regression model using the training set.
4. **Evaluation**:
   - Measures accuracy, sensitivity (true positive rate), and specificity (true negative rate) on the test set.
5. **Prediction**:
   - Takes user input for feature values and predicts the diabetes outcome using the trained model.

---

### **Evaluation Metrics**
- **Accuracy**: Measures the overall correctness of the model.
- **Sensitivity**: Proportion of actual positives correctly predicted (True Positive Rate).
- **Specificity**: Proportion of actual negatives correctly predicted (True Negative Rate).

---

### **File Descriptions**
- `CODE.py`: The main Python script containing the model implementation.
- `requirements.txt`: List of dependencies required to run the script.
- `data.csv`: Example dataset (not included, download the Pima Indians Diabetes Database).

---

### **Requirements**
- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `numpy`

---

### **Example**
**Input**:
```
Enter number of pregnancies: 2
Enter Glucose level: 85
Enter Blood Pressure: 66
Enter Skin Thickness: 29
Enter Insulin level: 125
Enter BMI: 26.6
Enter Diabetes Pedigree Function: 0.351
Enter Age: 31
```

**Output**:
```
Predicted outcome: Negative
with accuracy of 82.5% according to the Pima Indians Diabetes Database
```

---

### **References**
- Pima Indians Diabetes Database: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)
- Logistic Regression: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

