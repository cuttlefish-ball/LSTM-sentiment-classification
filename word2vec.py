import gensim.models
from utiles import  load_data
def text_to_vector(text,model):
    vector = [model[word] for word in text if word in model]
    return sum(vector) / len(vector) if vector else [0] * model.vector_size
def train_w2v(file_path):
    train=load_data(file_path+'/train/cut.txt')
    test = load_data(file_path+'/test/cut.txt')
    all=train+test
    all = [i for x in all for i in x]
    model = gensim.models.Word2Vec(sentences=all, vector_size=100, window=5, min_count=1, workers=5)
    model.wv.save_word2vec_format(file_path+'/w2v/word2vec.bin', binary=True)

def load_w2v(model_path,train_path,test_path):
    model=gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=True)
    train=load_data(train_path)
    test=load_data(test_path)
    train_w2v = [[text_to_vector(text,model)] for line in train for text in line]
    test_w2v = [[text_to_vector(text,model)] for line in test for text in line]
    return train_w2v,test_w2v,model.vector_size


if __name__ == '__main__':
    train_w2v('./data')