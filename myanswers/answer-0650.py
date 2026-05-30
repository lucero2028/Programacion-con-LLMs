import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def comparar_modelos_clasificacion(X, y):

    modelos = {
        "lr": LogisticRegression(),
        "dt": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier()
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        scores = cross_val_score(
            modelo,
            X,
            y,
            cv=5
        )

        resultados[nombre] = np.mean(scores)

    mejor = max(
        resultados,
        key=resultados.get
    )

    return (resultados, mejor)
