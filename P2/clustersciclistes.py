"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

def load_dataset(path):
	"""
	Carrega el dataset de registres dels ciclistes

	arguments:
		path -- dataset

	Returns: dataframe
	"""
	# TODO
	df = pd.read_csv(path)
	return df

def EDA(df):
	"""
	Exploratory Data Analysis del dataframe

	arguments:
		df -- dataframe

	Returns: None
	"""
	# TODO
	print(df.info())
	print(df.describe())
	print(df.isnull().sum())
	return None

def clean(df):
	"""
	Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

	arguments:
		df -- dataframe

	Returns: dataframe
	"""
	df = df.drop(columns=['id'])
	return df

def extract_true_labels(df):
	"""
	Guardem les etiquetes dels ciclistes (BEBB, ...)

	arguments:
		df -- dataframe

	Returns: numpy ndarray (true labels)
	"""
	# TODO
	true_labels = df['type'].values
	return true_labels



def visualitzar_pairplot(df):
	"""
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	arguments:
		df -- dataframe

	Returns: None
	"""
	# TODO
	sns.pairplot(df)
	plt.show()
	return None

def clustering_kmeans(data, n_clusters=4):
	"""
	Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	arguments:
		data -- les dades: tp i tb

	Returns: model (objecte KMeans)
	"""
	# TODO
	kmeans = KMeans(n_clusters=n_clusters)
	kmeans.fit(data)
	return kmeans

def visualitzar_clusters(data, labels):
	"""
	Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

	arguments:
		data -- el dataset sobre el qual hem entrenat
		labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

	Returns: None
	"""
	# TODO
	plt.figure(figsize=(10, 6))
	sns.scatterplot(data=data, x='tp', y='tb', hue=labels, palette='Set2')
	plt.title('Visualització dels Clústers')
	plt.show()
	return None

def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).

    arguments:
        tipus -- un array de tipus de patrons
        model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    dicc = {'tp': 0, 'tb': 1}
    # Associar clústers amb patrons
    centers = model.cluster_centers_
    suma_max = sum(centers[0])  # inicialització de suma màxima
    suma_min = sum(centers[0])  # inicialització de suma mínima
    ind_label_0 = ind_label_3 = -1

    for j, center in enumerate(centers):
        suma = sum(center)
        if suma > suma_max:
            suma_max = suma
            ind_label_3 = j
        if suma < suma_min:
            suma_min = suma
            ind_label_0 = j

    # Assignar la resta
    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)
    ind_label_1, ind_label_2 = lst

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})
    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    return tipus


def generar_informes(df, tipus):
	"""
	Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
	4 fitxers de ciclistes per cadascun dels clústers

	arguments:
		df -- dataframe
		tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

	Returns: None
	"""
	# TODO
	if not os.path.exists("informes"):
		os.makedirs("informes")
		
	for t in tipus:
		tipus_df = df[df['label'] == t['label']]
		filename = f'informes/{t["name"]}.txt'
		tipus_df.to_csv(filename, index=False)
    
	logging.info('S\'han generat els informes en la carpeta informes/\n')

	return None

def nova_prediccio(dades, model):
	"""
	Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

	arguments:
		dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
		model -- clustering model
	Returns: (dades agrupades, prediccions del model)
	"""
	# TODO
	dades = pd.DataFrame(dades, columns=['id', 'tp', 'tb', 'tt'])
	dades = dades.drop(columns=['id', 'tt'])
	prediccions = model.predict(dades)
	dades['label'] = prediccions
	return dades, prediccions

# ----------------------------------------------

if __name__ == "__main__":
    path_dataset = './P1/ciclistes.csv'

    df = load_dataset(path_dataset)
    EDA(df)
    df = clean(df)
    true_labels = extract_true_labels(df)

    # Visualitzar les dades
    visualitzar_pairplot(df)

    # Entrenar el model KMeans
    model = clustering_kmeans(df[['tp', 'tb']])

    # Guardar el model
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Calcular les mètriques
    homogeneity = homogeneity_score(true_labels, model.labels_)
    completeness = completeness_score(true_labels, model.labels_)
    v_measure = v_measure_score(true_labels, model.labels_)

    # Guardar les mètriques
    scores = {'homogeneity': homogeneity, 'completeness': completeness, 'v_measure': v_measure}
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump(scores, f)

    # Visualitzar els clusters
    visualitzar_clusters(df[['tp', 'tb']], model.labels_)

    # Afegir la columna label
    df['label'] = model.labels_

    # Associar clústers amb tipus
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
    tipus = associar_clusters_patrons(tipus, model)

    # Guardar l'assignació dels tipus
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus, f)

    # Generar informes
    generar_informes(df, tipus)

    # Classificació de nous ciclistes
    nous_ciclistes = [
        [500, 3230, 1430, 4670],  # BEBB
        [501, 3300, 2120, 5420],  # BEMB
        [502, 4010, 1510, 5520],  # MEBB
        [503, 4350, 2200, 6550]   # MEMB
    ]

    nouvelles, pred = nova_prediccio(nous_ciclistes, model)
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info(f'Tipus {i}: {t[0]["name"]} - Classe {p}')
