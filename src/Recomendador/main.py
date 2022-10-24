import pandas as pd 


def data():
    print("###Data()###")
    clients = pd.read_csv("../Data/archivos_auxiliares/clientes.csv", encoding="latin-1")
    clients_news = pd.read_csv("../Data/archivos_auxiliares/clientes_noticias.csv", encoding="latin-1") 
    news = pd.read_csv("../Data/archivos_auxiliares/noticias.csv", encoding="latin-1") 

    print(clients, clients.shape)
    print(clients_news, clients_news.shape)
    print(news, news.shape)
    
    return clients, clients_news, news 


if __name__ == '__main__':
    clients, clients_news, news = data() 
    print(clients, clients_news, news)
