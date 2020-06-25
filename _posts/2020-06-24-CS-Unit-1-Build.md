---
layout: post
title: K Nearest Neighbors Algorithm from Scratch
subtitle: How to build a K Nearest Neighbors Classifier in Python
gh-repo: dougscohen/cs-build-week-1
gh-badge: [star, fork, follow]
image: /img/doors_square.jpg
tags: [Data Science, Python, Alogrithm]
comments: true
---

## KNN Classifier from Scratch
In this tutorial, I will walk through how to create a K Nearest Neighbors algorithm in Python using just Numpy. KNNs are lazy algorithms. This essentially means they do not have a training phase, and instead the `fit()` method is meant to store the training data in memory to be accessed when you go to make a prediction. This also means that predictions can become computationally expensive as your data scales. 

Looking at a KNN is a great place to start studying classfier algorithms because it's straightforward, and pretty easy to understand what's happening behind the scenes. If you want to identify similar data, and you aren't necessarily concerned about efficiency, KNN may be the algorithm for you. Let's summarize the process:

1. Load in your data

2. Initiallize KNN class

3. Fit the training data

4. For each row in the training data:
    
    a. calculate euclidean distance between row and your query
    
    b. add the distances to a list
    
5. Sort the list from smallest distance to largest and grab the train indeces

6. Select the first K entries from the sorted list

7. Grab the class labels of the selected K entries

8. Return the class label that occured the most

Here, we fill focus on steps 2-8 as this is the actual algorithm.

## The Algorithm

You're probably wondering how the algorithm finds its "neighbors". In other words, what makes a row of data similar to other rows of data. For this model, I have used Euclidean distance to compare data. According to [Wikipedia](https://en.wikipedia.org/wiki/Euclidean_distance), it is the straight-line distance between two points. I will not go into the science behind Euclidean distance, so if you would like to further understand what is happening in this step, I recommend doing a litte research. However, let's take a look at the code and then walk through what's going on.

```python
def find_distance(self, row_A, row_B):
    
    # set distance to start at 0
    dist = 0.0
    # iterate through row_A
    for i in range(len(row_A)):
        # subtract the rows element wise, square the difference, then add
        #. it to the distance
        dist += (row_A[i] - row_B[i])**2

    # return the sqyare root of the total distance
    return np.sqrt(dist)
```

The comments will help walk you through the code, but let's take a look at it together. The function takes in two data points that are the same length. We want to iterate through each index of row_A, find its value, and subtract the value at the same index position in row_B. After squaring the difference, we then add that value to the existing distance total. Finally, we take the square root of the total distance to get the Euclidean distance between two points of data.

It's important to understand what is happening above, as it is vital in the prediction portion of a K Nearest Neighbors Classifier. In the next step, we will fit the KNN classifier. This step isn't necessary for a lazy learning algorithm, but it can be useful for storing the training data in memory. Let's take a look at what our KNN class would look like:

```python
class KNN():
    def __init__(self, K=5):
        self.K = K
        
    def fit(self, X_train, y_train=[]):
        self.X_train = X_train
        self.y_train = y_train
```

This shows that when we create a new class object, we choose the number of neighbors we want to find. It automatically defaults to 5. As mentioned above, when calling the `fit()` method, it will store the data to be used when we want to make a prediction. X_train represents a matrix of data which we will pull our neighbors from, and y_train contains the class labels for X_train. For example:

```python
X_train = [[1, 2, 3],        y_train = [0,
           [4, 5, 6],                   1,
           [7, 8, 9]]                   0]
```

The first row in X_train would belong to class "0". Second would belong to class "1". And third would belong to calss "0". This is extremely useful in testing the quality of the algorithm. You can split the data into testing and training, and then when you go to predict on the testing data, you can compare how well the predictions for reach row match up against the actual class for each row.

Now that we've covered how to find the Euclidean distance, and how to fit the classifier, let's get into the bulk of any KNN, the prediction. Say we have some data on sunflowers, and for each data point we know the petal length, petal width, and whether or not its a sunflower. Say we are now taking a walk and see a beautiful flower. We don't know whether it's a sunflower or not, but we are able to measure its petal length and width. We want to be able to make an educated guess about whether the flower is or is not a sunflower, so we plug our new data into the predicct method, to predict sunflower/not a sunflower. So let's define the predict method:

```python
def predict(self, X):

    # set predictinos to an empty list
    predictions = []

    # iterate (len(X)) number of times through 
    for i in range(len(X)):

      # list containing euclidean distances
      euclidean_distances = []

      # for each row in X_train, find its euclidean distance with the
      #. current 'X' row we are iterating through
      for row in self.X_train:
          eu_dist = self.find_distance(row, X[i])
          # append each euclidean distance to the list above
          euclidean_distances.append(eu_dist)

      # sort the euclidean distances from smallest to largest and grab
      #. the first K distances where K is the num_neigbors we want
      neighbor_indeces = np.array(euclidean_distances).argsort()[:self.K]

      # empty dictionary for class count
      neighbor_count = {}

      # for each neighbor, find its class
      for j in neighbor_indeces:
          if self.y_train[j] in neighbor_count:
              neighbor_count[self.y_train[j]] += 1
          else:
              neighbor_count[self.y_train[j]] = 1

      # get the most common class label and append it to predictions
      predictions.append(max(neighbor_count, key=neighbor_count.get))

    return predictions
```

So what's going on here? Basically we are iterating through X, where X can either be a single data point, or multiple data points. Each time through, we are going to calculate the Euclidian distance with every data point in X_train. However, we aren't concerned about every Euclidean distance, just the K smallest, where K is the number of neighbors. We grab the indeces and then find the class label for each neighbor (accessed through indeces in y_train). Out of all the neighbors, whichever class label appears the most is the one we will append to the predictions list.

So we've got methods to predict our target, but what if you just simply want to view neighbors. Let's create a method where you can input a single row of data, and return its nearest neighbors. Seeing neighbors should shed some light on how well our algorithm works. 

```python
def show_neighbors(self, x_instance):

    # list containing euclidean distances
    euclidean_distances = []

    # for each row in X_train, find its euclidean distance with the
    #. current 'X' row we are iterating through
    for row in self.X_train:
        eu_dist = self.find_distance(row, x_instance)
        # append each row and the euclidean distance to the list above
        euclidean_distances.append(eu_dist)

    # sort from smallest distance to largest distance and grab the first K
    #. indeces where K is the number of neigbors
    neighbor_indices = np.array(euclidean_distances).argsort()[:self.K]

    # list containg tuples of neighbor indeces and its euclidian distance
    #. to x_instance
    neighbors_and_distances = []

    for i in range(len(neighbor_indices)):
        val1 = neighbor_indices[i]
        val2 = euclidean_distances[i]
        neighbors_and_distances.append((val1, val2))
        
    return neighbors_and_distances
```

Finally, we have our code to return the k nearest neighbors. A lot of the steps are the same as the predict method, however this time, we return a list (of K length) of tuples, where each tuple contains a neighbor and its euclidean distance to the given input. Having a method that returns the neighbor indeces is great, because we can then go back in view each neighbor using said indeces.

## Compare

Okay. We have a working algorithm that was built from scratch. Let's see how well it performs compared to the K Nearest Neighbors classifier in Scikit-learn. 

### KNN Classifier from Scratch

```python
import pandas as pd
from sklearn.metrics import accuracy_score

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv('data/iris.csv', names=col_names)
train, test = train_test_split(iris, random_state=3)
X_train = train[column_names[:4]].values.tolist()
y_train = train['class'].values.tolist()
X_test = test[column_names[:4]].values.tolist()
y_test = test['class'].values.tolist()

knn = KNN(K=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"My model's accuracy: {accuracy_score(y_test, predictions)}")
```

KNN from scratch accuracy: **94.74%**

### KNN Classifier from Scikit-learn

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
predictions = neigh.predict(X_test)
print(f"Scikit-learn model's accuracy: {accuracy_score(y_test, predictions)}")
```
KNN from Scikit-learn accuracy: **94.74%**


EXACTLY THE SAME!


## Conclusion

A K Nearest Neighbors Classifier is one of the simpler classification algorithms out there. It's built on the premise that similar data points will appear close to each other in space. This makes it somewhat easy to interpret. Another advantage is no training phase. One major disadvantage is that KNNs become extremely slow as your data scales. It's best find another classifier when dealing with a large dataset. This tutorial shows an implementation of a KNN Classifier, however KNNs can work with regression problems as well. Instead of returning the most common class label of the resulting neighbors, you would return the average value of the class labels. 

