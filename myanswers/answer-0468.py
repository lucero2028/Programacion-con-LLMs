import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def estimar_eficiencia_termica(df_historico, condiciones_nuevas):

    X = df_historico[
        [
            'flujo_kg_s',
            'delta_t_entrada',
            'viscosidad_cp',
            'indice_ensuciamiento'
        ]
    ]

    y = df_historico['eficiencia_real']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = Ridge(alpha=1.0)
    modelo.fit(X_scaled, y)

    mae = mean_absolute_error(
        y,
        modelo.predict(X_scaled)
    )

    input_df = pd.DataFrame(
        [condiciones_nuevas],
        columns=X.columns
    )

    input_scaled = scaler.transform(input_df)

    prediccion = modelo.predict(input_scaled)[0]

    prediccion = max(0, min(100, prediccion))

    return {
        "eficiencia_estimada_porcentaje": round(float(prediccion), 2),
        "error_medio_modelo": round(float(mae), 3)
    }
