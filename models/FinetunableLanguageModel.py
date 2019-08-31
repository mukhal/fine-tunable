from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, TimeDistributed

from sklearn.datasets import make_classification
from FinetunableSequential import FinetunableSequentialClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

class FinetunableLanguageModel:

    """ 
    Defines a Finetunable Language Model based on the Keras Sequential API

    """

    def __init__(self, embedding_size=100, cell_type='LSTM', layer_size=100, n_layers=1, vocab_size=30000):
        
        self.embedding_size= embedding_size
        self.cell_type= cell_type
        self.layer_size = layer_size 
        self.n_layers = n_layers
        self.vocab_size = vocab_size + 1 # for pad symbol
        
        self._build_model(vocab_size)

    
    def _build_model(self, vocab_size):
        """
        builds the internal model based on FinetunableSequentialClassifier()

        Args:
            vocab_size (int): language model vocabulary size
        
        """

        self.model = FinetunableSequentialClassifier()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=self.embedding_size))
        
        if self.cell_type=='LSTM':
            rnn=LSTM
        elif self.cell_type=='GRU':
            rnn=GRU
        else :
            raise ValueError('Unknown %s RNN cell type' %(self.cell_type))
        
        for _ in range(self.n_layers):
            self.model.add(rnn(units=self.layer_size, return_sequences=True))

        
        self.model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
        self.model.summary()


    def train_on_texts(train_texts, epochs=10, optimizer='adam', lr=0.01):

        '''
        trains the language model on data

        Args: 
            train_texts (list of str): a list of sentences for training
            epochs (int): number of training epochs
            optimizer (str or ocject of type keras.optimizers)
        '''

        # tokenize text 
        tokenizer = Tokenizer(num_words=self.vocab_size)
        tokenizer.fit_on_texts(train_texts)
        self.tokenizer=tokenizer

        train_sequences = tokenizer.texts_to_sequences(train_texts)

        # find max sequence length
        max_seq_len = max([len(seq) for seq in train_sequences])
        
        max_seq_len = (max_seq_len // 2 + 1) * 2 # make sure it is even 
        max_seq_len =  min(max_seq_len, 256)
        
        # pad sequences
        train_sequences = pad_sequences(train_sequences, maxlen=max_seq_len, padding='post', value = self.vocab_size - 1)


    def train(train_ids, epochs=10):
        pass





       



if __name__ == '__main__':

    x = ['This is a sentence', 
    'I am a good programmer', 
    'this world is beautfully beautiful'
    ]

    model = FinetunableLanguageModel(n_layers=3)


