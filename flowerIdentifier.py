import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load dataset
from sklearn.datasets import load_iris
iris = load_iris()

#Convert to a dataframe for better readability
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target #Adding the species column (0,1,2)

#Convert numerical labels to actual species names
df['species'] = df['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

#Display the first 5 rows of the data set
print(df.head())

##Preprocessing the data
x = df.iloc[:, :-1] #Select all columns except the last one
y = df['species'] #Select only the species column
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,random_state=42)

#Standardize the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #Fit on training data
x_test = sc.transform(x_test) #Transform test data

#Training the model
model = KNeighborsClassifier(n_neighbors=5) # Create KNN model with k=5
model.fit(x_train, y_train) # Train the model on training data

y_pred = model.predict(x_test)

accuracy =accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy: .2f}')
print(classification_report(y_test,y_pred))

#Visualizing the data
sns.pairplot(df, hue="species", diag_kind="kde")
plt.show()