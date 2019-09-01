from finetunable.models.FinetunableLanguageModel import FinetunableLanguageModel




if __name__ == '__main__':

    x = ['This is a sentence', 
    'I am a good programmer', 
    'this world is beautfully beautiful'
    ]

    model = FinetunableLanguageModel(n_layers=1)
    
    model.train_on_texts(x, epochs=1)

    y= [1,0,1]

    model.finetune_for_clf(x,y,2)

