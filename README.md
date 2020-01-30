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