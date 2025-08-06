import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# Load the dataset
df = pd.read_csv('cleaned2.csv')
# Drop the unnamed column
df = df.drop(columns=['Unnamed: 0'])
# Define features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the classifier
rf.fit(X_train, y_train)
# Make predictions
y_pred = rf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
