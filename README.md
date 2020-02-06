# 1. Using the model:

### Setup

Make sure you have python3.x and pip3 installed after that just run 

```
source build_env
```

This will create a virtual environemnt named `env` and also install the requirements!

### Dataset Schema

| sentance_idx | word | tag  |
| ------------- |:-------------:| -----:|
| Sentance: 1    | philadelphia | B-fromloc.city_name | 
|    | to      |   O |
|  | san      |   B-toloc.city_name  |
|  | francisco      |   I-toloc.city_name  |

### Training & validation

```
python3 main.py --train
```

Validation is perfomed on every epoch, and based on the `F1-Score` the weights of the best model is saved in the `/trained_model` folder following the format - `trained_model_<N_EPOCHS>_<MODEL>.h5`.

### Metrics

Validation script in the `/metrics` folder, we're using the already exsiting `conlleval.pl` pearl script for finding out the `Precision`, `Recall` and `F1-Score` after every epoch.

:warning: NOTE :

The metrics works only for BIO tagged datasets.

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
Where is the stop for USAir flight number 37 from Philadelphia to San Francisco

37 - B-flight_number
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
    DROPOUT = 0.25
    N_EPOCHS = 20
    LOSS = crf_loss
    OPTIMIZER = 'rmsprop'
    MODEL = 'GRU_CRF'
    DATA_FILE = 'atis.pkl'
    EMBEDDINGS_FILE = 'word_embeddings.json'
    PORT = '5004'
```

`DATA_FILE` is to be saved in the `/data` folder and `EMBEDDINGS_FILE` gets automatically saved as per the name mentioned in the `Config` in the `/embeddings` folder.

### Logs

Once, the model is put into training, logs are generated and saved in `/logs` folder according to the format - `model_<N_EPOCHS>_<MODEL>.log` as mentioned in your configuration file. If you need different log outputs modify `/logs/logger.py`.

---

# 2. Using the API 
```
python3 app.py
```

Configure you app `PORT` in the configuration file, `model_config.py`

Go to your browser use the above example sentance:

Sentence:
`Show me all the nonstop flights between Atlanta and Philadelphia`

URL:
`http://localhost:8008/api/v1/slots/Show%20me%20all%20the%20nonstop%20flights%20between%20Atlanta%20and%20Philadelphia` 

API endpoint:

`/api/v1/slots/<sentance>`

Response: (JSON format)

```
{
  "response_time": "1.07s", 
  "sentance": "Show me all the nonstop flights between Atlanta and Philadelphia", 
  "slots": {
    "flight_stop": [
      "nonstop"
    ], 
    "fromloc.city_name": [
      "atlanta"
    ], 
    "toloc.city_name": [
      "philadelphia"
    ]
  }
}
```


