from flask import Flask, jsonify
from main import test
from model_config import Config
import tensorflow as tf
from process import Process

app = Flask(__name__)

def initApp():
    global process, graph

    process = Process()

    # Load trained model
    process.load('trained_model_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL))

    graph = tf.get_default_graph()

# Cross origin support
def sendResponse(responseObj):
    response = jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

@app.route('/api/v1/slots/<sentence>', methods=['GET'])
def main(sentence):
    with graph.as_default():
        response_time, slots = test(process, [sentence], read_file=False)

    response = {
        'sentance': str(sentence),
        'slots': slots,
        'response_time': str(response_time)[:4] + 's'
    }
    return sendResponse(response)

if __name__ == '__main__':
    '''
        Loading the model only once, instead of loading it on every request
    '''
    initApp()
    app.run(debug=True, port=Config.PORT, threaded=True)