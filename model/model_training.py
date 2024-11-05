from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

def generate_synthetic_data(num_samples, features):
    X = np.random.rand(num_samples, features)  
    y = np.random.randint(2, size=num_samples)  
    return X, y

num_samples = 1000   
features = 2          

X, y = generate_synthetic_data(num_samples, features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, '/home/kali/Downloads/NeuralGuard/model/saved_models/threat_detection_model.pkl')
print("Model saved to saved_models/threat_detection_model.pkl")
