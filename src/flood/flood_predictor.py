from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

file = (Path(__file__).parent / '../data/flood/flood_prediction_dataset.csv').resolve()
data = pd.read_csv(file)

data = data.drop(
    ['Sl', 'Station_Names', 'Year', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'Station_Number', 'Period'],
    axis=1
)

data['Flood?'] = data['Flood?'].fillna(0)

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

lr = LogisticRegression(max_iter=500)

# Accuracy: 94-95%
lr_clf = lr.fit(x_train, y_train)

# pickle.dump(lr_clf, open((Path(__file__).parent / '../models/lrclf.pkl').resolve(), 'wb'))
