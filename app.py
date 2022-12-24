import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask,request,render_template 

app=Flask(__name__) 

dataset=pd.read_csv('diabetes.csv')

X=dataset.iloc[:,:-1] 
X=X.values

Y=dataset.iloc[:,-1]
Y=Y.values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

rf_classifier=RandomForestClassifier()
rf_classifier.fit(X_train,Y_train)
rf_pred=rf_classifier.predict(X_test)
print(accuracy_score(rf_pred,Y_test)) 
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['post','get'])
def predict():
    pregnancies=request.form['pregnancies']
    glucose=request.form['glucose']
    bp=request.form['bp']
    st=request.form['st']
    insulin=request.form['insulin']
    bmi=request.form['bmi']
    dpf=request.form['dpf']
    age=request.form['age']
    data=[[pregnancies,glucose,bp,st,insulin,bmi,dpf,age]]
    outcome=rf_classifier.predict(data)
    if(outcome[0]==0):
        outcome='No Diabetes'
    else:
        outcome='Diabetes Found'
    return render_template('index.html',result=outcome)
if __name__=="__main__":
    app.run(port=5000,debug=True)