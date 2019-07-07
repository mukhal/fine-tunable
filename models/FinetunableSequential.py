from keras.models import Sequential
from keras.layers import Dense

from sklearn.datasets import make_classification

class FinetunableSequentialClassifier(Sequential):

    def __init__(self):
        super(FinetunableSequentialClassifier, self).__init__()

    
    def finetune (self, X=None, y=None, new_class_count=None, new_loss=None, layers_to_pop=1, new_optimizer=None, **kwargs):
        """finetunes an already trained model on the new_class_count"""

        assert new_class_count is not None, "please specify the target class count for finetuning..."
       
        # set new loss if any
        new_loss = new_loss or self.loss

        # set new optimizer if any
        new_optimizer = new_optimizer or self.optimizer
        # popping the last n layer
        for _ in range(layers_to_pop):
            self.pop()

        # add new classifieication head
        self.add(Dense(new_class_count, activation='softmax'))
        
        # compile new model with new loss
        self.compile(loss=new_loss, optimizer=new_optimizer, metrics=self.metrics)

        # finetune the new model
        self.fit(X, y, **kwargs) 


if __name__ =='__main__':

    X, y = make_classification()

    model = FinetunableSequentialClassifier()
    model.add(Dense(100, input_shape=(20,)))
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y)

    X, y = make_classification(n_classes=5, n_informative=10)

    model.finetune(X, y, new_class_count=5, new_loss ='sparse_categorical_crossentropy')


