# models/ml_model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class ConductivityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
