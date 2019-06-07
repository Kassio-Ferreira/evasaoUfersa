import pickle

import pandas as pd
import json
import numpy as np


def formata_objeto(x):
    objeto = pd.DataFrame(x['dados'])
    objeto = pd.get_dummies(objeto)

    df = objeto.reindex(columns=['reprovacoes', 'ira', 'acao_afirmativa_NAO', 'acao_afirmativa_SIM',
                                 'descricao_Noturno', 'descricao_Não Noturno', 'rede_ensino_Privada',
                                 'rede_ensino_Publica', 'sexo_F', 'sexo_M', 'PROAE_NAO', 'PROAE_SIM',
                                 'pesquisa_NÃO', 'pesquisa_SIM', 'estado_civil_Casado(a)',
                                 'estado_civil_NAO_INFORMADO', 'estado_civil_Solteiro(a)',
                                 'campus_ANGICOS', 'campus_CARAÚBAS', 'campus_MOSSORÓ',
                                 'campus_PAU DOS FERROS'], fill_value=0)

    df = np.array(df.values).reshape(len(df.index), 21)

    print(df)
    return df


def abre_logistica():
    pkl_filename = "model/reg_logistica.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def predicao_logistica(x, content):
    modelo = abre_logistica()
    predicao = modelo.predict_proba(x)
    matriculas = np.asarray(content['matriculas']).reshape(len(content['matriculas']), 1)
    predicao = np.column_stack((matriculas, predicao))
    print(predicao)
    keys = ["matricula", "prob_conclusao", "prob_abandono"]
    saida = [dict(zip(keys, values)) for values in predicao]
    return json.dumps(dict(resposta=saida), ensure_ascii=False)


def abre_tree():
    pkl_filename = "model/tree_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def predicao_tree(x, content):
    modelo = abre_tree()
    predicao = modelo.predict_proba(x)
    matriculas = np.asarray(content['matriculas']).reshape(3, 1)
    predicao = np.column_stack((matriculas, predicao))
    keys = ["matricula", "prob_conclusao", "prob_abandono"]
    saida = [dict(zip(keys, values)) for values in predicao]
    return json.dumps(dict(resposta=saida), ensure_ascii=False)


def abre_svm():
    pkl_filename = "model/svm.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def predicao_svm(x, content):
    modelo = abre_svm()
    predicao = modelo.predict_proba(x)
    matriculas = np.asarray(content['matriculas']).reshape(3, 1)
    predicao = np.column_stack((matriculas, predicao))
    keys = ["matricula", "prob_conclusao", "prob_abandono"]
    saida = [dict(zip(keys, values)) for values in predicao]
    return json.dumps(dict(resposta=saida), ensure_ascii=False)


def metodo_selec(x):
    objeto = x['metodo']
    print(objeto)
    return objeto


def seletor(content, objeto):
    metodo_selecionado = metodo_selec(content)
    if metodo_selecionado == 'logistic':
        output = predicao_logistica(objeto, content)
    elif metodo_selecionado == 'randomforest':
        output = predicao_tree(objeto, content)
    elif metodo_selecionado == 'svm':
        output = predicao_svm(objeto, content)
    else:
        return "error: unknown method."
    return output
