import pandas as pd 
import numpy as np
import nltk
from nltk import FreqDist
import re

import collections
from collections import Counter, OrderedDict
import torch
from torch import nn 
from torchtext.vocab import vocab

from TokenEmbedding import TokenEmbedding
from ToyNN import ToyNN

def clean_tokenizer_url(text):
    new_text = text.lower()
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    new_text = re.sub(regex , ' ', new_text)
    new_text = re.sub("\d+", ' ', new_text)
    new_text = re.sub("https?", ' ', new_text)
    new_text = re.sub("www?", ' ', new_text)
    new_text = re.sub("\\s+", ' ', new_text)
    new_text = new_text.split(sep = '-')
    
    new_text = [token for token in new_text if len(token) >2]
    
    return(new_text)
    

def clean_tokenizer_news(text):
    new_text = text.lower()
    new_text = re.sub('http\S+', ' ', new_text)
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    new_text = re.sub(regex , ' ', new_text)
    new_text = re.sub("\d+", ' ', new_text)
    new_text = re.sub("\\s+", ' ', new_text)
    new_text = new_text.split(sep = ' ')
    new_text = [token for token in new_text if len(token) > 2]

    return(new_text)


def tokenizer(db_news):
    textUrl=[]
    for i in range(0, db_news.shape[0]):
      textUrl.append(clean_tokenizer_url(db_news.news_url_absolute[i]))

    textTitle=[]
    for i in range(1, db_news.shape[0]):
      textTitle.append(clean_tokenizer_news(db_news.news_title[i]))

    textNews=[]
    for i in range(1, db_news.shape[0]):
      textNews.append(clean_tokenizer_news(db_news.news_text_content[i]))

    return textUrl, textTitle, textNews


def filter_country(db):
    textUrl=[]
    for i in range(1, db.shape[0]):
        url = clean_tokenizer_url(db.news_url_absolute[i])
        try:
            if url[1] == 'com' or url[1] == 'co':
                if url[2] == 'co' or url[2] not in country_iso:
                    textUrl.append(url)
        except IndexError:
            print("IndexError")
            print(url)
            textUrl.append(url)
            continue
    
    return textUrl

def acentuar(textTitle, textNews, TextUrl):
    for title in textTitle:
        for i in range(len(title)):
            if title[i] == "macroeconomia":
                title[i] = "macroeconomía"
        
            if title[i] == "reputacion":
                title[i] = "reputación"

            if title[i] == "innovacion":
                title[i] = "innovación"

            if title[i] == "economía":
                title[i] = "economía"
                    
        
    for new in textNews:
        for i in range(len(new)):
            if new[i] == "macroeconomia":
                new[i] = "macroeconomía"
        
            if new[i] == "reputacion":
                new[i] = "reputación"
        
            if new[i] == "innovacion":
                new[i] = "innovación"

            if new[i] == "economía":
                new[i] = "economía"

    for url in textUrl:
        for i in range(len(url)):
            if url[i] == "macroeconomia":
                url[i] = "macroeconomía"

            if url[i] == "reputacion":
                url[i] = "reputación"
        
            if url[i] == "innovacion":
               url[i] = "innovación"

            if url[i] == "economía":
               url[i] = "economía"
    return textUrl, textTitle, textNews

def post_training():
    print("Post_training")
    spanish_w2v = TokenEmbedding("../Data/archivos_auxiliares/SBW-vectors-300-min5.txt",500000)
    return spanish_w2v

def knn(W, x, k):
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')

def get_similar_tokens2(query_token, k, embed):
    W = embed.weight.data
    assert vocab_news.__contains__(query_token), "El token no está en el vocabulario"
    x = W[vocab_news[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    tokens = vocab_news.get_itos()
    arrayText=[]
    for i in topk[1:]:  # Remove the input words
        #print(f'cosine sim={float(cos[i]):.3f}: {tokens[i]}')
        arrayText.append(tokens[i])

    return(arrayText)

def make_vocab(oraciones,min_freq=1):
    #Comprueba que oraciones es una lista de listas
    if oraciones and isinstance(oraciones[0], list):
       #Transforma una lista anidada en una lista simple 
       tokens = [token for line in oraciones for token in line]
    counter_obj = collections.Counter()
    counter_obj.update(tokens)
    sorted_by_freq_tuples = sorted(counter_obj.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulario = vocab(ordered_dict, min_freq=min_freq)
    return vocabulario, ordered_dict

def create_dic():
    categorys = ["macroeconomía", "sostenibilidad", "innovación", "regulaciones", "alianzas", "reputación"]
    diccionary={}

    for category in categorys:
        k = get_similar_tokens2(category, 50, model.embedding)
        diccionary[category] = k

    return diccionary

def classifier_for_word_category(db_news, textUrl):
    data_count_categorys=db_news.copy()
    data_count_categorys[['news_text_content']]=data_count_categorys[['news_text_content']].astype(str)
    data_count_categorys['news_url_text']=pd.DataFrame(textUrl)

    print (data_count_categorys['news_url_text'])

    data_count_categorys2 = data_count_categorys.copy()
    for i in diccionary:
        exp_regular='\\b'+'(s)?(es)?\\b|\\b'.join(list(diccionary[i]))+'(s)?(es)?\\b'
        nomColum1='count_'+i+'_title'
        nomColum2='count_'+i+'_text'
        nomColum3='count_'+i+'_url'

        data_count_categorys2[nomColum1]=data_count_categorys2['news_title'].apply(lambda x: len(re.findall(exp_regular,x,re.IGNORECASE)))
        data_count_categorys2[nomColum2]=data_count_categorys2['news_text_content'].apply(lambda x: len(re.findall(exp_regular,x,re.IGNORECASE)))
        data_count_categorys2[nomColum3]=data_count_categorys2['news_url_text'].apply(lambda x: len(re.findall(exp_regular,x,re.IGNORECASE)))

    columnas2 = ['news_id',
           'count_macroeconomía_title', 'count_macroeconomía_text','count_macroeconomía_url',
           'count_sostenibilidad_title','count_sostenibilidad_text','count_sostenibilidad_url',
           'count_innovación_title', 'count_innovación_text', 'count_innovación_url',
           'count_regulaciones_title','count_regulaciones_text','count_regulaciones_url',
           'count_alianzas_title','count_alianzas_text', 'count_alianzas_url',
           'count_reputación_title','count_reputación_text','count_reputación_url']
    
    #print(data_count_categorys2[columnas2])
    return data_count_categorys, data_count_categorys2, columnas2


def classifier_for_clientCorporative(db_clients, db_clients_News, data_count_categorys, columnas2):
    data_result = pd.DataFrame()

    column_select = ['nit', 'news_id', 'news_url_absolute', 'news_init_date', 
        'news_final_date', 'news_title', 'count_name_tittle',
        'count_name_text','count_name_url', 'count_sect_tittle', 'count_sect_text','count_sect_url','count_ciiu_tittle', 'count_ciiu_text','count_ciiu_url'
           ]

    for j, i in enumerate(db_clients["nit"]):
            clie_news_filter = db_clients_news[db_clients_news["nit"]==i][["news_id", "nit"]]
            data_filter = pd.merge(data_count_categorys, clie_news_filter, on= "news_id")
            nombre = db_clients[db_clients["nit"]==i]["nombre"].iloc[0]
            sector = db_clients[db_clients["nit"]==i]["desc_ciiu_division"].iloc[0]
            ciiu = db_clients[db_clients["nit"]==i]["subsec"].iloc[0]
        
            data_filter["count_name_tittle"] = data_filter["news_title"].apply(lambda x: len(re.findall(nombre, x , re.IGNORECASE)))
            data_filter["count_name_text"] = data_filter["news_text_content"].apply(lambda x: len(re.findall(nombre, x, re.IGNORECASE)))
            data_filter["count_name_url"] = data_filter["news_url_text"].apply(lambda x: len(re.findall(nombre, x, re.IGNORECASE)))
            data_filter["count_sect_tittle"] = data_filter["news_title"].apply(lambda x: len(re.findall(sector, x, re.IGNORECASE)))
            data_filter["count_sect_text"] = data_filter["news_text_content"].apply(lambda x: len(re.findall(sector, x, re.IGNORECASE)))
            data_filter["count_sect_url"] = data_filter["news_url_text"].apply(lambda x: len(re.findall(sector, x, re.IGNORECASE)))
            data_filter["count_ciiu_tittle"] = data_filter["news_title"].apply(lambda x: len(re.findall(ciiu, x, re.IGNORECASE)))
            data_filter["count_ciiu_text"] = data_filter["news_text_content"].apply(lambda x: len(re.findall(ciiu, x, re.IGNORECASE)))
            data_filter["count_ciiu_url"] = data_filter["news_url_text"].apply(lambda x: len(re.findall(ciiu, x, re.IGNORECASE)))
        
            data_result = pd.concat([data_result,data_filter[column_select]], ignore_index=True)
            #print("Termine de contar el cliente:" + str(i) + " avance " + str((j + 1)/len(db_clients["nit"])))
    print(data_result)
    return data_result


def classifier(data_result_final):
    nit=[]
    id_new=[]
    clasification=[]
    particip=[]
    db_teamName = [] 
    for i in range(0, data_result_final.shape[0]):
        db_teamName.append("UN_Data")
        id_new.append(data_result_final.news_id[i])
        nit.append(data_result_final.nit[i])
        sum_macroeconomia, sum_sostenibilidad, sum_innovacion, sum_regulacion, sum_alianzas,  sum_reputacion = 0, 0, 0, 0, 0, 0
        count=[]
        sum_macroeconomia=data_result_final.count_macroeconomía_text[i]+2*data_result_final.count_macroeconomía_title[i]+4*data_result_final.count_macroeconomía_url[i]          
        sum_sostenibilidad=data_result_final.count_sostenibilidad_text[i]+2*data_result_final.count_sostenibilidad_title[i]+4*data_result_final.count_sostenibilidad_url[i]
        sum_innovacion=data_result_final.count_innovación_text[i]+2*data_result_final.count_innovación_title[i]+4*data_result_final.count_innovación_url[i]
        sum_regulacion=data_result_final.count_regulaciones_text[i]+2*data_result_final.count_regulaciones_title[i]+data_result_final.count_regulaciones_url[i]
        sum_alianzas=data_result_final.count_alianzas_text[i]+2*data_result_final.count_alianzas_title[i]+4*data_result_final.count_alianzas_url[i]
        sum_reputacion=data_result_final.count_reputación_text[i]+2*data_result_final.count_reputación_title[i]+4*data_result_final.count_reputación_url[i]
    
        count += [sum_macroeconomia, sum_sostenibilidad, sum_innovacion, sum_regulacion, sum_alianzas,  sum_reputacion]
        max_value = max(count) 
        index=count.index(max_value)
    
        if max_value==0:
            clasification.append('Descartable')
        else:
            if index==0:
               clasification.append('macroeconomía')              
            if index==1:
               clasification.append('sostenibilidad')
            if index==2:
               clasification.append('innovacion')
            if index==3:
               clasification.append('regulaciones')
            if index==4:
               clasification.append('alianzas')
            if index==5:
               clasification.append('reputacion')

        if data_result_final.count_name_tittle[i]>0 or data_result_final.count_name_text[i] >0 or data_result_final.count_name_url[i] >0 :       
            particip.append('Cliente')
        elif data_result_final.count_sect_tittle[i]>0 or data_result_final.count_sect_text[i] >0 or data_result_final.count_sect_url[i] >0 :       
            particip.append('Sector') 
        elif data_result_final.count_ciiu_tittle[i]>0 or data_result_final.count_ciiu_text[i]>0 or data_result_final.count_ciiu_url[i] >0 :       
            particip.append('Sector') 
        else:
            particip.append('No Aplica')  

    print("vars")
    print(len(db_teamName))
    print(len(nit))
    print(len(id_new))
    print(len(particip))
    print(len(clasification))

    db_clasifier2 = pd.DataFrame()
    db_clasifier2['UN_Data'] = db_teamName
    db_clasifier2['nit'] = nit
    db_clasifier2['id_new'] = id_new
    db_clasifier2['participacion'] = particip
    db_clasifier2['recomendacion'] = clasification
    
    db_clasifier2.to_csv("../Data/Output/recomendacion.csv", index=False)

def data():
    db_clients = pd.read_csv("../Data/archivos_auxiliares/clientes.csv", encoding="latin-1")
    db_clients_news = pd.read_csv("../Data/archivos_auxiliares/clientes_noticias.csv", encoding="latin-1") 
    db_news = pd.read_csv("../Data/archivos_auxiliares/noticias.csv", encoding="latin-1") 

    db_clients["desc_ciiu_division"] = db_clients["desc_ciiu_division"].apply(lambda x: re.sub(r'[^\w\s]', "",x.lower()))
    db_clients["desc_ciiuu_clase"] = db_clients["desc_ciiuu_clase"].apply(lambda x: re.sub(r'[^\w\s]', "",x.lower()))
    db_clients["desc_ciuu_grupo"] = db_clients["desc_ciuu_grupo"].apply(lambda x: re.sub(r'[^\w\s]', "",x.lower()))
    db_clients["subsec"] = db_clients["subsec"].apply(lambda x: re.sub(r'[^\w\s]', "",x.lower()))

    db_news["news_title"] = db_news["news_title"].apply(lambda x: re.sub(r'[^\w\s]', "",x.lower()))
    db_news["news_text_content"] = db_news["news_text_content"].apply(lambda x: re.sub(r'[^\w\s]', "",x.lower()))

    country_iso = pd.read_csv("../Data/archivos_auxiliares/country_iso.csv", sep=',')
    country_iso = list(country_iso[' iso2'].str.lower())
    country_iso.remove('co')


    return db_clients, db_clients_news, db_news, country_iso


if __name__ == '__main__':
    db_clients, db_clients_news, db_news, db_country_iso = data() 

    #print(clients.head(), clients_news.head(), news.head())
    #print(country_iso)
    #print(filter_country(news))

    textUrl, textTitle, textNews = tokenizer(db_news)
    textUrl, textTitle, textNews = acentuar(textUrl, textTitle, textNews)
    spanish_w2v = post_training() 
    #get_similar_tokens('reputación', 100, spanish_w2v) # test
    vocab_news, ordered_dict = make_vocab(textNews,5)
    #print(vocab_news, ordered_dict)

    matrix_len = len(vocab_news.get_itos())
    weights_matrix = np.zeros((len(vocab_news.get_itos()), spanish_w2v.dim))
    words_found = 0
    words_not_found = []

    for i, word in enumerate(vocab_news.get_itos()):
        try: 
            weights_matrix[i] = spanish_w2v[[word]][0]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(spanish_w2v.dim, ))
            words_not_found.append(word)

    #print(weights_matrix.shape)

    model = ToyNN(weights_matrix,256,2)
    #print(model)
    
    #k=get_similar_tokens2('macroeconomía', 200, model.embedding)
    #print(k)
    
    diccionary = create_dic()
    #print(diccionary)
    data_count_categorys, data_count_categorys2, columnas2 = classifier_for_word_category(db_news, textUrl)
    data_result = classifier_for_clientCorporative(db_clients, db_clients_news, data_count_categorys, columnas2)

    data_result2 = pd.merge(db_clients_news[["nit", "news_id"]], data_count_categorys2[columnas2], on = "news_id", how = "left")
    data_result_final_aux = pd.merge(data_result, data_result2, on = ["nit", "news_id"])
    data_result_final = pd.merge(db_clients[["nit", "nombre"]], data_result_final_aux, on = "nit")
    
    print(data_result_final)
    classifier(data_result_final)