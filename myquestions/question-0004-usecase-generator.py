Generador de casos de uso
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_detectar_anomalias_isolation_forest():
    np.random.seed()
    
    # Datos normales
    X_normal = np.random.normal(loc=0, scale=1, size=(25, 3))
    
    # Datos anómalos (valores extremos)
    X_anomalias = np.random.normal(loc=8, scale=1, size=(5, 3))
    
    # Combinar
    X = np.vstack([X_normal, X_anomalias])
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelo
    modelo = IsolationForest(contamination=0.15)
    modelo.fit(X_scaled)
    
    etiquetas = modelo.predict(X_scaled)
    num_anomalias = np.sum(etiquetas == -1)
    
    return ({"X": X, "contamination": 0.15}, (etiquetas, num_anomalias))

i, o = generar_caso_de_uso_detectar_anomalias_isolation_forest()

print('---- inputs ----')
for k, v in i.items():
    print("\n", k, ":\n", v)

print('\n---- expected output ----\n', o)
