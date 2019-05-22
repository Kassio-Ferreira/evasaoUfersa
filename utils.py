import pickle

import pandas as pd
import json
import numpy as np


def formata_objeto(x):
    objeto = pd.DataFrame(x, columns=x.keys(), index=[0])
    del objeto['metodo']
    print(objeto)
    objeto = pd.get_dummies(objeto)

    df = objeto.reindex(columns=['reprovacoes', 'ira', 'acao_afirmativa_NAO', 'acao_afirmativa_SIM',
                   'descricao_Noturno', 'descricao_Não Noturno', 'rede_ensino_Privada',
                   'rede_ensino_Publica', 'sexo_F', 'sexo_M', 'PROAE_NAO', 'PROAE_SIM',
                   'pesquisa_NÃO', 'pesquisa_SIM', 'estado_civil_Casado(a)',
                   'estado_civil_NAO_INFORMADO', 'estado_civil_Solteiro(a)',
                   'campus_ANGICOS', 'campus_CARAÚBAS', 'campus_MOSSORÓ',
                   'campus_PAU DOS FERROS'], fill_value=0)
    print(df)
    df = np.array(df.iloc[0].values).reshape(1, 21)
    print(df)
    return df


def abre_logistica():
    pkl_filename = "model/reg_logistica.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def predicao_logistica(x):
    modelo = abre_logistica()
    predicao = modelo.predict_proba(x)
    print(modelo.classes_)
    print(modelo.predict(x))
    return json.dumps(dict(prob_conclusao=predicao[0, 0], prob_abandono=predicao[0, 1]),
                      ensure_ascii=False)


def abre_tree():
    pkl_filename = "model/tree_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def predicao_tree(x):
    modelo = abre_tree()
    predicao = modelo.predict_proba(x)
    print(modelo.classes_)
    print(modelo.predict(x))
    return json.dumps(dict(prob_conclusao=predicao[0, 0], prob_abandono=predicao[0, 1]),
                      ensure_ascii=False)


def abre_svm():
    pkl_filename = "model/svm.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def predicao_svm(x):
    modelo = abre_svm()
    predicao = modelo.predict_proba(x)
    print(modelo.classes_)
    print(modelo.predict(x))
    return json.dumps(dict(prob_conclusao=predicao[0, 0], prob_abandono=predicao[0, 1]),
                      ensure_ascii=False)


def metodo_selec(x):
    objeto = pd.DataFrame(x, columns=x.keys(), index=[0])
    return objeto['metodo'][0]


def seletor(content, objeto):
    metodo_selecionado = metodo_selec(content)
    if metodo_selecionado == 'logistic':
        output = predicao_logistica(objeto)
    elif metodo_selecionado == 'randomforest':
        output = predicao_tree(objeto)
    elif metodo_selecionado == 'svm':
        output = predicao_svm(objeto)
    else:
        return "error: unknown method."
    return output
