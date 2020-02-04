# Slot-filling

### Setup

Make sure you have python3.x and pip3 installed after that just run 

```
source build_env
```

This will create a virtual environemnt named `env` and also install the requirements!

### Training & validation

```
python3 main.py --train
```

Validation is perfomed on every epoch, and based on the `F1-Score` the weights of the best model is saved in the `/trained_model` folder following the format - `trained_model_<N_EPOCHS>_<MODEL>_weights.h5` and it's corresponding JSON as `trained_model_<N_EPOCHS>_<MODEL>.json`.

### Metrics

Validation script in the `/metrics` folder, we're using the already exsiting `conlleval.pl` pearl script for finding out the `Precision`, `Recall` and `F1-Score` after every epoch.

### Testing

You can directly test using the above command as the repo includes the `trained_model` as well :D

Go to `tests/text_sentances.txt` :

```
I want to see all the flights from washington to berlin flying tomorrow



```

Make sure you add every sentance on one line and the run :

```
python3 main.py --test
```

Go to `tests/slots.txt` to find the output :

```
I want to see all the flights from washington to berlin flying tomorrow

washington - B-fromloc.city_name
berlin - B-toloc.airport_code
tomorrow - B-depart_date.today_relative
----------------------------------------------------------------------------------------------------


```

```
NOTE:

Above example was the output when tested on the ATIS dataset and 
might not work with the current fork of the code, as it is still in development.
```

### Configuration

Your model can be easily configured in the file `model_config.py` 

Ex:

```
class Config:
    EMBEDDING_SIZE = 100
    HIDDEN_UNITS = 100
    DROPOUT = 0.5
    N_EPOCHS = 20
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'rmsprop'
    MODEL = 'BLSTM'
    DATA_FILE = 'ner_dataset.csv'
    EMBEDDINGS_FILE = 'word_embeddings.json'
```

`DATA_FILE` is to be saved in the `/data` folder and `EMBEDDINGS_FILE` gets automatically saved as per the name mentioned in the `Config` in the `/embeddings` folder.

### Logs

Once, the model is put into training, logs are generated and saved in `/logs` folder according to the format - `model_<N_EPOCHS>_<MODEL>.log` as mentioned in your configuration file. If you need different log outputs modify `/logs/logger.py`.