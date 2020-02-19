import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

df = pd.read_csv("online_shoppers_intention.csv")

colummnLabel = preprocessing.LabelEncoder();

weekends = colummnLabel.fit_transform(list(df["Weekend"]))
month = colummnLabel.fit_transform(list(df["Month"]))
visitorType = colummnLabel.fit_transform(list(df["VisitorType"]))
rev = colummnLabel.fit_transform(list(df["Revenue"]))

relatedProduct = list(df["ProductRelated"])
relatedProductDuration = list(df["ProductRelated_Duration"])
bounceRates = list(df["BounceRates"])
exitRates = list(df["ExitRates"])
browser = list(df["Browser"])
region = list(df["Region"])
trafficType = list(df["TrafficType"])
specialDay = list(df["SpecialDay"])
info = list(df["Informational"])
infoDuration = list(df["Informational_Duration"])

x = list(zip(relatedProduct, relatedProductDuration, bounceRates, exitRates, region, trafficType, weekends, month, visitorType, info, infoDuration))
#df = df[["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "OperatingSystems", "Browser", "Region", "TrafficType"]]

y = list(rev)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

def myLinearModel():
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linearMddel = linear_model.LinearRegression()
    linearMddel.fit(x_train, y_train)
    acc = linearMddel.score(x_test, y_test)
    print(acc)

def myNeighborsModel():
    best_score = 0.85

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    '''
        best neighbor fit = 13
    '''
    neighborsModel = KNeighborsClassifier(n_neighbors=13)
    neighborsModel.fit(x_train, y_train)

    acc = neighborsModel.score(x_test, y_test)
    if acc > best_score:
        print(acc)
        with open("buyer-intensions.pickle", "wb") as f:
            pickle.dump(neighborsModel, f)


def runModel():
    buyer_intentions_model = open("buyer-intensions.pickle", "rb")
    trainedModel = pickle.load(buyer_intentions_model)
    prediction = trainedModel.predict(x_test)

    values = ["True", "False"]

    '''
        plt.plot(bounceRates, month)
        plt.show()
    '''
    for x in range(len(x_test)):
        print("Prediciton: ", values[prediction[x]], "Data: ", x_test[x], "Actual Results: ", values[y_test[x]])


'''for _ in range(100):
    myNeighborsModel()'''

runModel()
