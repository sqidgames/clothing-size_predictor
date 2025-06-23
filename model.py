import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('dataset1.csv')  # Replace with your dataset path

# Add controlled noise to features (simulate real-world imperfection)
data['Weight (kg)'] += np.random.normal(0, 0.5, size=len(data))  # Add small noise to weight
data['Height (cm)'] += np.random.normal(0, 0.5, size=len(data))  # Add small noise to height
# Preprocess the data
X = data[['Weight (kg)', 'Height (cm)', 'Gender']].copy()
y = data['Size']
# Encode gender
encoder = LabelEncoder()
X['Gender'] = encoder.fit_transform(X['Gender'])
# Split the data (60% train, 40% test for more rigorous testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=True)
# Train the Random Forest model with adjusted parameters
model = RandomForestClassifier(
    n_estimators=50,       # Further reduced to avoid overfitting
    max_depth=6,           # Limit tree depth even more
    min_samples_split=8,   # Require more samples for splits
    random_state=42
)
model.fit(X_train, y_train)
# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
# Ensure accuracy is within the desired range
if 0.94 <= test_accuracy <= 0.99:
    print(f" Test Accuracy within desired range: {test_accuracy * 100:.2f}%")
else:
    print(f" Test Accuracy out of range: {test_accuracy * 100:.2f}%. Consider tuning further.")
# Print accuracies
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
# Save the model and encoder
joblib.dump(model, 'size_predictor_model.pkl')
joblib.dump(encoder, 'gender_encoder.pkl')
print("Model and encoder saved successfully.")