from flask import Flask, request, jsonify
import numpy as np
import pickle as p
from keras import backend as K


app = Flask(__name__)


def load_model():

    global model
    model = None
    # model variable refers to the global variable
    with open('gait_predictor.pickle', 'rb') as f:
        model = p.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    # jdata = request.get_json()
    # print('Data at Server:', jdata)
    # data = pd.read_json(json.dumps(jdata), orient='records')
    # print("After conversion: ", data)
    load_model()
    with open('xtest.pkl', 'rb') as f:
        data = p.load(f)
    predictions = np.array2string(model.predict_classes(data[0:1]))
    K.clear_session()
    print(jsonify(predictions))
    return jsonify(predictions)


print("Server is getting ready...")
app.run(host='0.0.0.0', debug=True)


