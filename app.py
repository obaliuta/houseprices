
"""
App of predicting housing prices in California

"""


import numpy as np
from flask import Flask, request, render_template
import pickle

#creating app object with a Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))


#using the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


#Redirect to /predict page with the output
@app.route('/predict',methods=['POST'])
def predict():
    background="background.jpg"
    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted House Value is {}'.format(output),background=background)


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()