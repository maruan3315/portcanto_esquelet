"""
Este módulo genera un dataset de ciclistas.
"""

import logging
import random
import pandas as pd

def generar_dataset(num, ind, dicc):
    """
    Genera els temps dels ciclistes, de forma aleatòria, 
    però en base a la informació del diccionari.
    num: número de files/ciclistes a generar.
    ind: index/identificador/dorsal.
    dicc: diccionari amb la informació dels ciclistes.
    """
    data = []
    for i in range(num):
        ciclista_type = random.choice(dicc)
        ciclista = {
            'id': ind + i,
            'type': ciclista_type['name'],
            'temps_pujada': random.gauss(ciclista_type['mu_p'], ciclista_type['sigma']),
            'temps_baixada': random.gauss(ciclista_type['mu_b'], ciclista_type['sigma'])
        }
        data.append(ciclista)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    SRT_CICLISTES = 'data/ciclistes.csv'

    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240  # mitjana temps pujada bons escaladors
    MU_P_ME = 4268  # mitjana temps pujada mals escaladors
    MU_B_BB = 1440  # mitjana temps baixada bons baixadors
    MU_B_MB = 2160  # mitjana temps baixada mals baixadors
    SIGMA = 240     # 240 s = 4 min

    dicc2 = [
        {"name":"BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name":"MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    dataset = generar_dataset(10, 1, dicc2)
    dataset.to_csv(SRT_CICLISTES, index=False)
    logging.info("s'ha generat ciclistes.csv")
