from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from bitarray import test
import pandas as pd
import numpy as np

def open_file_path():
    # global file_path
    file_path = filedialog.askopenfilename()
    inputtxt1.insert(1.0, file_path)
    inputtxt1.grid(column=1, row=0)

def run_code():
    file_path = inputtxt1.get(1.0, "end-1c")
    parameters = params.get(1.0, "end-1c")
    prediction = predict.get(1.0, "end-1c")

    if file_path != "" and parameters != "":

        df = pd.read_csv(file_path)
        cdf = df
        msk = np.random.rand(len(df)) < 0.8
        train = cdf[msk]
        test = cdf[~msk]
        # 'file_path' is the file location as string
        # 'parameters' is a string of parameters
        # Use parameters = parameters.split(",") to get the list of parameters
        # 'result' is the variable string

        parameters = parameters.split(",")
        prediction = prediction.split(",")



        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        x_train = np.asanyarray(train[parameters])
        y_train = np.asanyarray(train[prediction[0]])
        regr.fit(x_train, y_train)
        # predictions
        y_hat = regr.predict(test[parameters])
        x_test = np.asanyarray(test[parameters])
        y_test = np.asanyarray(test[prediction[0]])
        # error
        error1 = np.mean((y_hat - y_test) ** 2)



        from sklearn.tree import DecisionTreeRegressor

        # Fit regression model
        regr = DecisionTreeRegressor(max_depth=1)
        regr.fit(x_train, y_train)
        # Predict
        y_hat = regr.predict(x_test)
        # error
        error2 = np.mean((y_hat - y_test) ** 2)



        from sklearn.ensemble import RandomForestRegressor

        regr = RandomForestRegressor(max_depth=1, random_state=0)
        regr.fit(x_train, y_train.ravel())
        # Predict
        y_hat = regr.predict(x_test)
        # error
        error3 = np.mean((y_hat - y_test) ** 2)


       # Output to be seen by user

        result = "Multiple Linear Regression: \n\t\tResidual sum of squares:\t" + f"{error1}"\
                 "\n\nDecision Tree Regression: \n\t\tResidual sum of squares:\t" + f"{error2}"\
                 "\n\nRandom Forest Regressor: \n\t\tResudual sum of Squares:\t" + f"{error3}"
        result_box = Text(window, height=20, width=100, padx=10, pady=5)
        result_box.insert(1.0, result)
        result_box.grid(columnspan=3, row=4)


    else:
        text_box = Text(window, height=1, width=50, padx=10, pady=5)
        inp = "Please check file and parameters."
        text_box.insert(1.0, inp)
        text_box.grid(columnspan=3, row=4)


if __name__ == "__main__":
    window = Tk()
    window.title("Model")

    canvas = Canvas(window,bg = "#FFFFFF",height = 300,width = 600)
    canvas.grid(columnspan=3,rowspan=3)

    p = Label(window,text="Enter File Path(csv):",font=("Raleway",12))
    p.grid(columnspan=1,column=0,row=0)

    inputtxt1 = Text(window,height = 1,width = 30)
    inputtxt1.grid(column=1,row=0)

    Button(window, text="Browse", command=open_file_path).grid(column=2,row=0)

    p = Label(window,text="Enter Parameters:",font=("Raleway",12))
    p.grid(columnspan=1,column=0,row=1)

    params = Text(window,height = 1,width = 30)
    params.grid(column=1,row=1)

    Button(window, text="Run",command=run_code).grid(column=2,row=1)
    # canvas = Canvas(window,bg = "#FFFFFF",height = 250,width = 600)
    # canvas.grid(columnspan=3)

    p = Label(window,text="Enter prediction: ",font=("Raleway",12))
    p.grid(columnspan=1,column=0,row=2)

    predict = Text(window,height = 1,width = 30)
    predict.grid(column=1,row=2)

    window.mainloop()