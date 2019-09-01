# Finetunable



Finetunable is an NLP transfer learning package based on Keras. You can pre-train a model for language modeling and finetune it on your target task in a few lines of code. 

## Usage


### Pretraining

The main class is the `FinetunableLanguageModel` class.

```python
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
```python
# create  classification dataset
x = ['this is so sad',
     'I am so happy']
# labels
y = [0, 1]

# finetune
model.finetune_for_clf(x,y,2)
```


## TODO

* Language Model finetuning as in [ULMFit](https://arxiv.org/abs/1801.06146)
* Discriminative Learning Rates
* Gradual Unfreezing
* Masked loss to avoid cmoputing loss for the pad symbol
* Transformer Models in addition to RNN



## License

MIT 

