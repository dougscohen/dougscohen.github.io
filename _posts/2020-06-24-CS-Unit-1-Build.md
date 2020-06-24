---
layout: post
title: K Nearest Neighbors Algorithm from Scratch
subtitle: How to build a K Nearest Neighbors Classifier in Python
gh-repo: dougscohen/cs-build-week-1
gh-badge: [star, fork, follow]
tags: [Data Science, Python, Alogrithm]
comments: false
---

## KNN Classifier from Scratch
In this tutorial, I will walk through how to create a K Nearest Neighbors algorithm in Python using just Numpy. KNNs are lazy algorithms. This essentially means they do not have a training phase, and instead the `fit()` method is meant to store the training data to be accessed when you go to make a prediction.

Looking at a KNN is a great place to start studying classfier algorithms because it's straightforward, and pretty easy to understand what's happening behind the scenes. Let's dive into it!

## The Algorithm

You're probably wondering how the algorithm finds its "neighbors". In other words, what makes a row of data similar to other rows of data. For this model, I have used Euclidean distance to compare data. According to [Wikipedia](https://en.wikipedia.org/wiki/Euclidean_distance), it is the straight-line distance between two points. I will not go into the science behind Euclidean distance, so if you would like to further understand what is happening in this step, I recommend doing a litte research. However, let's take a look at the code and then walk through what's going on.

```python
def find_distance(self, row_A, row_B):
    """
    Returns Euclidean distance between 2 rows of data.

        Parameters:
                row_A (list): vector of numerical data
                row_B (list): vector of numerical data

        Returns:
             Euclidean Distance (float)
    """
    
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

The comments will help walk you through the code, but let's take a look at it together. The function takes in 2 rows of data that are the same length. Assuming each row has a length greater than 1, we want to iterate through each index of row_A, find its value, and subtract the value at the same index position in row_B. After squaring the difference, we then add that value to the existing distance total. Finally, we take the square root of the total distance to get the Euclidean distance between two rows of data.

It;s important to understand what is happening above, as it is vital in the prediction portion of a K Nearest Neighbors Classifier. In the next step, we will fit the KNN classifier. This step isn't necessary for a lazy learning algorithm, but it can be useful for storing the training data in memory. Let's take a look at what our KNN class would like like:

```python
class KNN():
    
    def __init__(self, num_neighbors=5):
        """
        Initialization of algorithm
        """
        self.num_neighbors = num_neighbors
        
    def fit(self, X_train, y_train):
        """
        Fits the model to the training data
        """
        self.X_train = X_train
        self.y_train = y_train
```

This shows that when we create a new class object, we choose the number of neighbors we want to find. It automatically defaults to 5. As mentioned above, when calling the `fit()` method, it will store the data to be used when we want to make a prediction. X_train represents a matrix of data which we will pull our neighbors from, and y_train are the target values for X_train. For example:

```python
X_train = [[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]]

y_train = [0, 1, 0]
```

The first row in X_train would belong to class "0". Second would belong to class "1". And third would belong to calss "O". This is extremely useful in testing how useful our alogrithm is. You can split the data into testing and training, and then when you go to predict on the testing data, you can compare how well the predictions for reach row mtach up against the actual class for each row.

Now that we've covered how to find the Euclidean distance, and how to fit the classifier, let's get into the bulk of any KNN, the prediction. This is the point of it, after all. Say we have a flower, and we know it's characteristics. We want to be able to make an educated guess about the type of flower it is. We use the data we already have on hand to predict the target (flower). So let's define the predict method:

```python
def predict(self, X):
      """
      Returns a list of predictions for the inputed X matrix.

      Parameters:
              X (list): 2D list/array of numerical values

      Returns:
              predictions (list): list of predictions for each of the vectors
              in X. 
      """
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
          euclidean_distances_sorted = np.array(euclidean_distances).argsort()[:self.num_neighbors]

          # empty dictionary for class count
          neighbor_count = {}
          
          # for each neighbor, find its class
          for j in euclidean_distances_sorted:
              if self.y_train[j] in neighbor_count:
                  neighbor_count[self.y_train[j]] += 1
              else:
                  neighbor_count[self.y_train[j]] = 1

          # get the most common class label and append it to predictions
          predictions.append(max(neighbor_count, key=neighbor_count.get))

      return predictions
```

So what's going on here?






Here's a useless table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |


How about a yummy crepe?

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)

It can also be centered!

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg){: .center-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
