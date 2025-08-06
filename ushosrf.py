# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
# Step 2: Load your dataset
df = pd.read_csv('cleaned1.csv')
# Print column names to verify
print("Column names in the dataset before get_dummies:")
print(df.columns)

# Step 3: Preprocess your data
# Convert categorical features to numerical using get_dummies
df = pd.get_dummies(df)
# Print column names again after get_dummies
print("Column names in the dataset after get_dummies:")
print(df.columns)
# Combine 'readmitted' columns into a single binary column
df['readmitted'] = df[['readmitted_<30', 'readmitted_>30']].max(axis=1)
# Drop the original 'readmitted' columns
df.drop(columns=['readmitted_<30', 'readmitted_>30', 'readmitted_NO'], inplace=True)
# Step 4: Separate features and target variable
X = df.drop(columns='readmitted')
y = df['readmitted']
# Fill missing values with mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 6: Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# Step 7: Make predictions and evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
