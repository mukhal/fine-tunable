from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, TimeDistributed, Lambda

from sklearn.datasets import make_classification
from FinetunableSequential import FinetunableSequentialClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np
import logging

logger = logging.getLogger(__file__)
logging.basicConfig(level='INFO')

class FinetunableLanguageModel:

    """ 
    Defines a Finetunable Language Model based on the Keras Sequential API
    """

    def __init__(self, embedding_size=100, cell_type='LSTM', layer_size=100, n_layers=1, vocab_size=30000):
        
        self.embedding_size= embedding_size
        self.cell_type= cell_type
        self.layer_size = layer_size 
        self.n_layers = n_layers
        self.vocab_size = vocab_size 
        self.padding_index = self.vocab_size
        self.is_pretrained= False

        logger.info('Building internal Language Model')
        self._build_model(self.vocab_size + 1) # +1 for pad symbol

    
    def _build_model(self, vocab_size):
        
        """
        builds the internal model based on FinetunableSequentialClassifier()
        Args:
            vocab_size (int): language model vocabulary size
        """

        self.model = Sequential()
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


    def train_on_texts(self, train_texts, epochs=10, max_seq_len=256, batch_size=1, optimizer='adam'):

        '''
        trains the language model on data

        Args: 
            train_texts (list of str): a list of sentences for training
            epochs (int): number of training epochs
            optimizer (str or ocject of type keras.optimizers)
        '''

        # tokenize text 
        logger.info('Tokenizing texts...')
        tokenizer = Tokenizer(num_words=self.vocab_size)
        tokenizer.fit_on_texts(train_texts)

           
        logger.info('Using a vocabulary size of %d' %(len(tokenizer.word_index)))
        self.tokenizer=tokenizer

        train_sequences = tokenizer.texts_to_sequences(train_texts)

        # find max sequence length
        max_len = max([len(seq) for seq in train_sequences])
        
        max_len = (max_len // 2 + 1) * 2 # make sure it is even 
        max_len =  min(max_len, max_seq_len)
        

        logger.info('Creating training sequences of maximum len %d' %(max_len))

        # pad sequences
        train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', value = self.padding_index)

        # create (x, y) train and label sequences
        x, y = train_sequences[:, :-1], train_sequences[:, 1:]
        assert(x.shape == y.shape)

        # train the language model
        self.pretrain(x, y, epochs, batch_size, optimizer)


    def pretrain(self, x, y, epochs, batch_size, optimizer):
        """
        pretrain the language model given the above parameters

        Args:
            x (numpy array of shape (D, seq_len)): input sequences
            y (numpy array of shape (D, seq_len)): next word sequences

        """
        # compile internal model
        # TODO: add perplexity metric
        self.model.compile(loss = 'sparse_categorical_crossentropy', optimizer= optimizer) 
        
        # reshape y to (D, seq_len , 1)
        y = y[:, :, np.newaxis]
        
        # train model
        self.model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size)
        self.is_pretrained= True


    def finetune_for_clf(self, X=None, y=None, new_class_count=None, new_loss=None, new_optimizer=None, **kwargs):
        """
        finetunes the pretrained LM for classification

        TODO: discriminative learning rate
        TODO: gradual unfreezing
        
        """
        assert self.is_pretrained, "You must pretrain the language model first before finetuning it!"
        assert new_class_count is not None, "please specify the target class count for finetuning!"
        assert new_loss is not None, "please specify the new finetuning loss!"
       
        # set new optimizer if any
        new_optimizer = new_optimizer or self.optimizer
        
        # popping the last layer
        self.pop()

        # adding maxpooling layer
        maxpool = lambda
        
        # add new classifieication head
        self.add(Dense(new_class_count, activation='softmax'))
        
        # compile new model with new loss
        self.compile(loss=new_loss, optimizer=new_optimizer, metrics=self.metrics)

        # finetune the new model
        self.fit(X, y, **kwargs) 








    



        






       



if __name__ == '__main__':

    x = ['This is a sentence', 
    'I am a good programmer', 
    'this world is beautfully beautiful'
    ]

    model = FinetunableLanguageModel(n_layers=1)
    
    model.train_on_texts(x)


