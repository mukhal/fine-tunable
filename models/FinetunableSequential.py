from keras.models import Sequential


class FinetunableSequential(Sequential):


    def finetune(X=X_train, y=y_train, validation_data=None, gradual_unreezing=False, Slanted_Triangular_Learning_rate=False):

        pass