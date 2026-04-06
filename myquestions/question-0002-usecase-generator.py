
 Generador
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def generar_caso_de_uso_comparar_modelos_clasificacion():
    np.random.seed()
    
    X = np.random.rand(40, 4)
    y = np.random.randint(0, 2, 40)
    
    modelos = {
        "lr": LogisticRegression(),
        "dt": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier()
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=5)
        resultados[nombre] = np.mean(scores)
    
    mejor = max(resultados, key=resultados.get)
    
    return ({"X": X, "y": y}, (resultados, mejor))
    
i, o = generar_caso_de_uso_comparar_modelos_clasificacion()

print('---- inputs ----')
for k, v in i.items():
    print("\n", k, ":\n", v)

print('\n---- expected output ----\n', o)
