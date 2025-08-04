import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned data
df = pd.read_csv(r"C:\Users\Tejas\OneDrive\Desktop\glioma_grading\data\glicomaxl\cleaned_data.csv")

# Prepare features (X) and target (y)
X = df.drop(['Grade', 'Case_ID', 'Project', 'Primary_Diagnosis', 'Age_at_diagnosis'], axis=1)
y = df['Grade']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, r"C:\Users\Tejas\OneDrive\Desktop\glioma_grading\data\glicomaxl\rf_model.pkl")
print("Model saved to rf_model.pkl")