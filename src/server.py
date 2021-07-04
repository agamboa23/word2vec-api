import flask
from flask import request, jsonify
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import json
import numpy
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from flask_cors import CORS, cross_origin

books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]



app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.config["DEBUG"] = True
wv_from_bin=''

@app.route("/")
@cross_origin()
def test():
    message=    {'message': 'Test'}
    return jsonify(message)

@app.route("/loadModel")
@cross_origin()
def loadModel():
    print("Loading model")
    global wv_from_bin
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/Users/alexis/Documents/Thesis/code/word2vec-service/data/GN.bin"), binary=True)
    print("Model loaded")
    return jsonify({'Message':'Model loaded'})


@app.route('/testSim', methods=['GET'])
@cross_origin()
def home():
    global wv_from_bin
    model = wv_from_bin
    distance = model.distance("media", "medium")
    return jsonify(distance)

@app.route('/word2vec', methods=['POST'])
@cross_origin()
def word2vec():
    global wv_from_bin
    model = wv_from_bin
    words = request.form['words']
    words=words.split(',')
    
    res = ((x, model[x]) if x in model else None for x in words)
    result = list(res)

    if words[0] in model:
        return jsonify({ 'data': model[words] })

    return jsonify({'message': '404 Not Found'})

@app.route('/getUserCentroids', methods=['POST'])
@cross_origin()
def getUserCentroids():
    labels = request.form['words']
    labels = labels.split(',')
    k = request.form['k']
    global wv_from_bin
    model = wv_from_bin
    res = map(lambda x:model[x] if x in model else None,labels)
    result = list (res)
    clean = [x for x in result if x is not None]
    X = numpy.array(clean)
    kmeans = KMeans(n_clusters=int(k),random_state=0)
    print(X.shape)
    kmeans.fit(X)
    return jsonify(kmeans.cluster_centers_.tolist())

@app.route('/getPlaceScore', methods=['POST'])
@cross_origin()
def getPlaceScore():
    global wv_from_bin
    model = wv_from_bin
    labels = request.form['words']
    labels=labels.split(',')
    userCentroids = request.form['centroids']
    userCentroids = numpy.array(json.loads(userCentroids))
    res = map(lambda x:model[x] if x in model else None,labels)
    result = list (res)
    clean = [x for x in result if x is not None]
    if len(clean)==0:
        scores = numpy.ones(16)*-1
        return jsonify(scores.tolist())    
    X = numpy.array(clean)
    placeVector = numpy.mean(X,axis=0)
    print(placeVector)

    distances = numpy.array([])
    for centroid in userCentroids:
        singleDistance = numpy.linalg.norm(centroid-placeVector)
        distances=numpy.append(distances,singleDistance)
    minim = distances.min() if distances.size>0 else 0
    meany = distances.mean() if distances.size>0 else 0
    return jsonify({'scores':distances.tolist(),'min':minim, 'mean':meany})

app.run()

