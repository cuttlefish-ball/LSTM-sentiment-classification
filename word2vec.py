from gensim.models import Word2Vec, word2vec
from utiles import  load_data
def text_to_vector(text,model):
    vector = [model.wv[word] for word in text if word in model.wv]
    return sum(vector) / len(vector) if vector else [0] * model.vector_size
def train_w2v(file_path):
    train=load_data(file_path+'/train/cut.txt')
    test = load_data(file_path+'/test/cut.txt')
    all=train+test
    all = [i for x in all for i in x]
    model = Word2Vec(sentences=all, vector_size=100, window=5, min_count=1, workers=5)
    model.save(file_path+'/w2v/word2vec.model')

def load_w2v(model_path,train_path,test_path):
    model=Word2Vec.load(model_path)
    train=load_data(train_path)
    test=load_data(test_path)
    train_w2v = [[text_to_vector(text,model)] for line in train for text in line]
    test_w2v = [[text_to_vector(text,model)] for line in test for text in line]
    return train_w2v,test_w2v


if __name__ == '__main__':
    train_w2v('data')