import pandas as pd
import numpy as np

def generar_caso_de_uso_procesar_datos_transporte():
    np.random.seed()
    
    df = pd.DataFrame({
        "distancia": np.random.randint(1, 100, 30),
        "tiempo": np.random.randint(1, 10, 30),
        "costo": np.random.randint(50, 500, 30)
    })
    
    df["target"] = df["costo"]
    
    df["costo_por_km"] = df["costo"] / df["distancia"]
    
    corr = df.corr()
    columnas = corr["target"].abs() > 0.3
    
    esperado = df.loc[:, columnas]
    
    return ({"df": df, "target_col": "target"}, esperado)

i, o = generar_caso_de_uso_procesar_datos_transporte()

print('---- inputs ----')
for k, v in i.items():
    print("\n", k, ":\n", v)

print('\n---- expected output ----\n', o)
