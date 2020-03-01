from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import json

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the review comments in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            filename = 'xgboost_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))
            prediction = loaded_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
            print('prediction is', prediction)
            if(prediction[0]==0):
                predicted_class='Setosa'
            elif((prediction[0]==1)):
                predicted_class = 'Versicolour'
            else :
                predicted_class = 'Virginica'
            return render_template('results.html',predicted_class=predicted_class)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
	#app.run(debug=True)