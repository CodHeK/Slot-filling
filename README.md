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

Validation is perfomed on every epoch, and based on the `F1-Score` the weights of the best model is saved in the `/trained_model` folder following the format - `trained_model_<N_EPOCHS>_<MODEL>.h5`.

### Metrics

Validation script in the `/metrics` folder, we're using the already exsiting `conlleval.pl` pearl script for finding out the `Precision`, `Recall` and `F1-Score` after every epoch.

### Testing

You can directly test using the above command as the repo includes the `trained_model` as well :D

Go to `tests/text_sentances.txt` :

```
Where is the stop for USAir flight number thirty-seven from Philadelphia to San Francisco?

```

Make sure you add every sentance on one line and the run :

```
python3 main.py --test
```

Go to `tests/slots.txt` to find the output :

```
Where is the stop for USAir flight number 37 from Philadelphia to San Francisco?

37 - B-airline_code
philadelphia - B-fromloc.city_name
san - B-toloc.city_name
francisco - I-toloc.city_name
----------------------------------------------------------------------------------------------------

```

:warning: NOTE :

This project is still in development and the results might not be very accurate at the moment.

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