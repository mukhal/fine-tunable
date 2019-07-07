from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, TimeDistributed

from sklearn.datasets import make_classification

class FinetunableLanguageModel:

    def __init__(self, embedding_size=100, cell_type='LSTM', layer_size=100, n_layers=1):
        
        self.embedding_size= 100
        self.cell_type= cell_type
        self.layer_size = layer_size
        self.n_layers = 1

    
    def _build_model(self, vocab_size, optmizer='adam', lr=0.001):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=self.embedding_size))
        
        if self.cell_type=='LSTM':
            rnn=LSTM
        elif self.cell_type=='GRU':
            rnn=GRU
        else :
            raise ValueError('Unknnown %s RNN cell type' %(self.cell_type))
        
        for _ in range(self.n_layers):
            model.add(rnn(units=self.layer_size, return_sequences=True))

        model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
        
        model.summary()
        self.model=model

    def train_on_text(data, epochs=10, lr=0.01):

        '''
        data: a list of strings representing a set of sentences
        '''

        ## tokenize 
        #   



if __name__ == '__main__':

    X, y = make_classification()

    model = FinetunableLanguageModel(n_layers=3)


