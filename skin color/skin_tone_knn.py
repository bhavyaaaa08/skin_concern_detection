import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def skin_tone_knn(mean_values):
    df = pd.read_csv("/content/drive/MyDrive/your_folder/skin_tone_dataset.csv")
    
    X = df.iloc[:, [1, 2, 3]].values
    y = df.iloc[:, 0].values
    
    classifier = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
    classifier.fit(X, y)
    
    X_test = [mean_values]
    y_pred = classifier.predict(X_test)
    return y_pred[0]