from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU

from sklearn.datasets import make_classification

class FinetunableLanguageModel:

    def __init__(self, vocab_size=1000, embedding_size=100, n_layers=1, cell_type='LSTM'):
        super(FinetunableLanguageModel, self).__init__()
        
        model = Sequential()
        model.add()
        self.add()
    
    def finetune (self,
            X=None,
            y=None,
            new_class_count=None,
            
            **kwargs):

        assert new_class_count is not None, "please specify the target class count for finetuning..."
        # extract keyword arguments
        new_loss = kwargs['new_loss'] if 'new_loss' in kwargs else self.loss
        kwargs.pop('new_loss')
        
        # popping the last layer
        self.pop()

        self.add(Dense(new_class_count, activation='softmax'))
        self.compile(loss=new_loss, optimizer=self.optimizer, metrics=self.metrics)

        self.fit(X, y, **kwargs) 


if __name__ =='__main__':

    X, y = make_classification()

    model = FinetunableSequential()
    model.add(Dense(100, input_shape=(20,)))
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y)

    X, y = make_classification(n_classes=5, n_informative=10)

    model.finetune(X, y, new_class_count=5, new_loss ='sparse_categorical_crossentropy')


