import pandas as pd 
import numpy as np
import nltk
from nltk import FreqDist
import re

import collections
from collections import Counter, OrderedDict
import torch
from torchtext.vocab import vocab

from TokenEmbedding import TokenEmbedding

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


def data():
    db_clients = pd.read_csv("../Data/archivos_auxiliares/clientes.csv", encoding="latin-1")
    db_clients_news = pd.read_csv("../Data/archivos_auxiliares/clientes_noticias.csv", encoding="latin-1") 
    db_news = pd.read_csv("../Data/archivos_auxiliares/noticias.csv", encoding="latin-1") 

    country_iso = pd.read_csv("../Data/archivos_auxiliares/country_iso.csv", sep=',')
    country_iso = list(country_iso[' iso2'].str.lower())
    country_iso.remove('co')


    return db_clients, db_clients_news, db_news, country_iso


if __name__ == '__main__':
    clients, clients_news, news, country_iso = data() 

    #print(clients.head(), clients_news.head(), news.head())
    #print(country_iso)
    #print(filter_country(news))

    #print(tokenizer(news))
    spanish_w2v = post_training() 
    get_similar_tokens('reputación', 100, spanish_w2v)
    #195
