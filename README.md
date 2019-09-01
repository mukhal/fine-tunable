# Fineutnable



Finetunable is an NLP transfer learning package based on Keras. You can pre-train a model for language modeling and finetune it on your target task in a few lines of code. 

## Usage


### Pretraining

The main class is the `FinetunableLanguageModel` class.

```
from finetunable.models.FinetunableLanguageModel import FinetunableLanguageModel # import model

# sample pretraining dataset
x = ['This is a sentence', 
    'Finetuable is great', 
    'this world is beautfully beautiful'
    ]
    
# create your Language model

model = FinetunableLanguageModel(n_layers=3, cell_type='GRU', layer_size=100, vocab_size=50)

# pretraing the language model 
model.train_on_texts(x, epochs=1)

```


### Finetuning for classfication

To finetune the model, you need to call `model.finetune_for_clf()` and pass the classfication dataset and the new class count :
```
# create  classification dataset
x = ['this is so sad',
     'I am so happy']

# labels
y = [0, 1]

# finetune
model.finetune_for_clf(x,y,2)

    
```
