import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import utiles

def process_data(save_path,start,num,csv_path):

    df=pd.read_csv(csv_path,encoding='utf-8',encoding_errors='ignore',header=None)

    pos=df.loc[start+8e5:start+8e5+num-1,5]
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
    print(train)
    utiles.save_data(train,file_path+'/cut.txt')

def le_words(file_path):
    all=utiles.load_data(file_path+'/cut.txt')
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [[lemmatizer.lemmatize(word.lower()) for word in words] for words in all]
    utiles.save_le_data(lemmatized_words,file_path+'/le_words.txt')


if __name__ == '__main__':
    csv_path = 'data/raw_data.csv'
    data_path='./data'
    process_data(data_path+'/train', 0, 1e5,csv_path)
    process_data(data_path+'/test',11e5,2000,csv_path)
    cut_words(data_path+'/train')
    cut_words(data_path+'/test')
    le_words(data_path+'/train')
    le_words(data_path + '/test')