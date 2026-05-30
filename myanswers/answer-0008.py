import numpy as np
from sklearn.ensemble import RandomForestClassifier


def predecir_resistencia(X_train, y_train, X_test):

    modelo = RandomForestClassifier(random_state=42)

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    return predicciones
