import pandas as pd

def clasificar_base_lubricante(muestras):

    clasificaciones = []

    for _, fila in muestras.iterrows():

        if (
            fila['viscosity_index'] > 150
            and
            fila['flash_point'] > 230
        ):
            clasificaciones.append('Sintético')
        else:
            clasificaciones.append('Mineral')

    return clasificaciones
