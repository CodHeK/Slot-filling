from flask import Flask, jsonify, request
from main import test
from model_config import Config
import tensorflow as tf
from process import Process
from utils.files import getBestSavedModel
from logs.logger import log

app = Flask(__name__)

def initApp():
    global process, graph

    process = Process()

    _, best_model_filename, _ = getBestSavedModel()

    log('Best model : ' + str(best_model_filename) + ' picked for loading!')

    # Load trained model
    process.load(best_model_filename)

    graph = tf.get_default_graph()

# Cross origin support
def sendResponse(responseObj):
    response = jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

'''
Ex:
    http://localhost:9009/api/v1/slots?sentence=
    Where%20is%20the%20stop%20for%20USAir%20flight%20number%2037%20from%20
    Philadelphia%20to%20San%20Francisco%20flying%20next%20friday
'''
@app.route('/api/v1/slots', methods=['GET'])
def main():
    sentence = str(request.args.get('sentence'))
    with graph.as_default():
        response_time, slots = test(process, [sentence], read_file=False)

    response = {
        'sentence': sentence,
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