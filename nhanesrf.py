import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
file_path = '/content/random.csv'
df = pd.read_csv(file_path)

# Impute all numerical columns with mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical variables (if any)
# Assuming 'Gender' is categorical
df['Gender'] = df['Gender'].astype('category').cat.codes

# Convert 'Breast_fed' to categorical if it is not already
df['GlycoHemoglobin'] = df['GlycoHemoglobin'].astype('int')

# Ensure there are no missing values left
assert df.isnull().sum().sum() == 0, "There are still missing values in the dataset"

# Split the data into features and target
# Assuming 'Breast_fed' is the target variable
X = df.drop(columns=['GlycoHemoglobin'])
y = df['GlycoHemoglobin']

# Check unique values in the target to ensure it's categorical
print(y.unique())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

