from keras.models import Sequential
from keras.layers import Dense

class FinetunableSequential(Sequential):


    def finetune (self,
            x=None,
            y=None,
            new_class_count=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            slanted_triangular_learning_rates=False,
            discriminative_finetuning=False,
            gradual_unfreezing=False,

            **kwargs):

        assert new_class_count is not None

        self.pop()
        # popping the last layer 
        self.add(Dense(2, activation='softmax'))
        self.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.fit(X,y) 


if __name__ =='__main__':

    pass