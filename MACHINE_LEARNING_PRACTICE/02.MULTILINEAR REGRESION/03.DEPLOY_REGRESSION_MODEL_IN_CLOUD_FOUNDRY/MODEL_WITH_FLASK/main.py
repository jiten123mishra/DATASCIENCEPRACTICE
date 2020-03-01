ws  1232azfrom flask import Flask, request
from flask import Response
import os 

from linear_regression_predict import getPrediction
from linear_regression_train import model_train

app = Flask(__name__)
#CLUD DEPLOYMENT SPECIFIC
os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')


@app.route("/predict", methods=['GET'])
def predictRoute():
    val1 = request.args.get("val1")
    val2 = request.args.get("val2")
    val3 = request.args.get("val3")
    modelFileName = "finalized_model_advertisement.sav"
    result = getPrediction(modelFileName, val1, val2, val3)
    return Response(str(result))


@app.route("/train", methods=['POST'])
def trainModel():
    modelFileName = "finalized_model_advertisement.sav"
    dataset = "Advertising.csv"
    model_train(dataset, modelFileName)
    return Response("Success")
    
#LOCAL SPECIFIC
#if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=5000)
    
#CLOUD DEPLOYMENT SPECIFIC
port=int(os.getenv("PORT"))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
