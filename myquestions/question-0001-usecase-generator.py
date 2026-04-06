Generador:
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_pipeline_churn_clientes():
    np.random.seed()
    
    df = pd.DataFrame({
        "edad": np.random.randint(18, 70, 30),
        "ingresos": np.random.randint(1000, 5000, 30),
        "genero": np.random.choice(["M", "F"], 30),
        "ciudad": np.random.choice(["A", "B", "C"], 30),
        "churn": np.random.randint(0, 2, 30)
    })
    
    X = pd.get_dummies(df.drop("churn", axis=1))
    y = df["churn"]
    
    modelo = RandomForestClassifier()
    modelo.fit(X, y)
    acc = accuracy_score(y, modelo.predict(X))
    
    return ({"df": df, "target_col": "churn"}, acc)

i, o = generar_caso_de_uso_pipeline_churn_clientes()

print('---- inputs ----')
for k, v in i.items():
    print("\n", k, ":\n", v)

print('\n---- expected output ----\n', o)
