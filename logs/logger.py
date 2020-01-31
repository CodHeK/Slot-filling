import logging

def log(message):
    LOG_FORMAT = '%(asctime)s - %(message)s'
    logging.basicConfig(filename='logs/model.log', 
                        filemode='w',
                        level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt='%d-%b-%y %H:%M:%S')

    logging.info(message)
