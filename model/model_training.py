from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Generate synthetic data
def generate_synthetic_data(num_samples, features):
    X = np.random.rand(num_samples, features)  # Random feature values
    y = np.random.randint(2, size=num_samples)  # Binary target (0 or 1)
    return X, y

# Parameters
num_samples = 1000    # Number of synthetic samples
features = 2          # Number of features (protocol and flags)

# Generate synthetic training data
X, y = generate_synthetic_data(num_samples, features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, '/home/kali/Downloads/NeuralGuard/model/saved_models/threat_detection_model.pkl')
print("Model saved to saved_models/threat_detection_model.pkl")
