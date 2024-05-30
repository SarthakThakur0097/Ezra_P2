import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import numpy as np
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Your code here
def evaluate_preds(y_true, y_preds):
    """
    Perform evaluation comparison on y_true labels vs. y_pred labels on a classification
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {'accuracy': round(accuracy, 2),
                  'precision': round(precision, 2),
                  'recall': round(recall, 2),
                  'f1': round(f1, 2)}
    print(f'Acc: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}%')
    print(f'Recall: {recall:.2f}%')
    print(f'F1 score: {f1:.2f}%')
    
    return metric_dict

# Load your data into a pandas DataFrame
data = pd.read_csv("test_data.csv")

# Define features (X) and target variable (y)
X = data.drop(columns=["Repurchase", "DOB"])  # Drop non-numeric and target columns
y = data["Repurchase"]

# Define categorical columns
categorical_cols = ["City", "State", "Type_of_Scan", "UTM", "Facility"]

# Define numerical columns
numerical_cols = ["Age", "Times_Bought", "Time_Intervals"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess categorical variables with one-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]).toarray(), columns=encoder.get_feature_names_out(categorical_cols))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]).toarray(), columns=encoder.get_feature_names_out(categorical_cols))

# Normalize numerical variables
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_cols]), columns=numerical_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_cols]), columns=numerical_cols)

# Concatenate encoded categorical and scaled numerical features
X_train_processed = pd.concat([X_train_encoded, X_train_scaled], axis=1)
X_test_processed = pd.concat([X_test_encoded, X_test_scaled], axis=1)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train_processed, y_train)
# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X_train_processed, y_train, cv=5)  # cv=5 means 5-fold cross-validation

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Calculate and print the average score
print("Average accuracy:", np.mean(cv_scores))
# Predict on the test set
y_pred = rf_classifier.predict(X_test_processed)

evaluate_preds(y_test, y_pred)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
##print("Accuracy:", accuracy)

