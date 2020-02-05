from flask import Flask, jsonify
from main import test
from model_config import Config

app = Flask(__name__)

@app.route('/api/v1/slots/<sentence>', methods=['GET'])
def main(sentence):
    slots = test([sentence], read_file=False)
    response = {
        'sentance': str(sentence),
        'slots': slots
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=Config.PORT)