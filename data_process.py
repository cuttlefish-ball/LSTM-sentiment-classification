import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import utiles

def process_data(save_path,start,num):
    csv_path='data/raw_data.csv'
    df=pd.read_csv(csv_path,encoding='utf-8',encoding_errors='ignore',header=None)

    pos=df.loc[start+80000:start+80000+num-1,5]
    neg=df.loc[start:start+num-1,5]


    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    pos = pos.apply(lambda s: remove_pattern(s,"@[\w]*"))
    neg = neg.apply(lambda s: remove_pattern(s,"@[\w]*"))

    pos = pos.apply(lambda s: re.sub(r'[^\w\s]', '', s))
    neg = neg.apply(lambda s: re.sub(r'[^\w\s]', '', s))

    pos.to_csv(save_path+'/pos_data.csv',index=False,header=False)
    neg.to_csv(save_path+'/neg_data.csv',index=False,header=False)

def cut_words(file_path):
    pos=utiles.load_data(file_path+'/pos_data.csv')
    neg = utiles.load_data(file_path+'/neg_data.csv')
    train=pos+neg
    for s in train:
        s[0]=nltk.tokenize.word_tokenize(s[0])
    stop_words = stopwords.words("english")
    train=[w for w in train if not w in stop_words]
    utiles.save_data(train,file_path+'/cut.txt')

if __name__ == '__main__':
    process_data('./data/train', 0, 10000)
    process_data('./data/test',20000,2000)
    cut_words('data/train')
    cut_words('data/test')