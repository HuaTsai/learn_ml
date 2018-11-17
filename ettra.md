Errata
- 24 submitted: last submission 20 Sep 2018 

Chapter 2, Training Machine Learning Algorithms for Classification | Section: Training a perceptron model on the Iris dataset | Page no: 28

One-vs.-All (OvA), or sometimes also called One-vs.-Rest (OvR), is a technique, us to extend a binary classifier to multi-class problems.

Should be:

One-vs.-All (OvA), or sometimes also called One-vs.-Rest (OvR), is a technique, used to extend a binary classifier to multi-class problems.

Chapter 1, Giving Computers the Ability to Learn from Data | Section: Solving interactive problems with reinforcement learning | Page no: 6

The labells "Action" and "State" have been swapped in the diagram.

Chapter 1, Giving Computers the Ability to Learn from Data | Section: An introduction to the basic terminology and notations | Page no: 8

The Iris dataset contains the measurements of 150 iris flowers from three different species: Setosa, Versicolor, and Viriginica.

Should be :

The Iris dataset contains the measurements of 150 iris flowers from three different species: Setosa, Versicolor, and Virginica. 


Chapter 1, Giving Computers the Ability to Learn from Data | Section: An introduction to the basic terminology and notations | Page no: 10

Thus, each row in this feature matrix represents one flower instance and can be written as four-dimensional column vector

Should be:

Thus, each row in this feature matrix represents one flower instance and can be written as four-dimensional row vector

And 

Each feature dimension is a 150-dimensional row vector

Should be:

Each feature dimension is a 150-dimensional column vector

Errata Type: Support Query | Access to code bundle and other data sets |


The datasets to GitHub, the reader can find the datasets at https://github.com/rasbt/python-machine-learning-book/tree/master/code/datasets

The direct link to the Iris dataset would be: https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data

We've added some additional notes to the code notebooks mentioning the offline datasets in case there are server errors. https://www.dropbox.com/sh/tq2qdh0oqfgsktq/AADIt7esnbiWLOQODn5q_7Dta?dl=0

Errata Type: Typo l Page No. 116 

This: Since we the fit the LogisticRegression object on a multiclass dataset, it uses the

Should be:  Since we fit the LogisticRegression object on a multiclass dataset, it uses the

Errata Type: Typo l  Page No. 128

This: In a nutshell, PCA aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions that the original one.

Should be:  In a nutshell,PCA aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions than the original one.

Errata Type: Code, Page number 205.

It is:

p(i_1 | x)=0.2x0.1 + 0.2x0.2 + 0.6x0.06

Should be:

p(i_1 | x)=0.2x0.1 + 0.2x0.2 + 0.6x0.6

Errata Type: Grammar | Chapter 11 | Page 313

It is: 'Thus, our goal is to group the samples based on their feature similarities, which we can be achieved using the k-means algorithm...'

Should be: 'Thus, our goal is to group the samples based on their feature similarities, which can be achieved using the k-means algorithm...'

Errata Type: Code | Chapter 10 | Page 298

It is:  y=w0 + w1x + w2x2 x2+…+wdxd

Should be:  y=w0 + w1x + w2x2+…+wdxd

Errata Type: Code| Chapter 10 | Page 305

It is: >>> sort_idx = X.flatten().argsort()

                  >>> lin_regplot(X[sort_idx], y[sort_idx], tree)

Should be: >>> sort_idx = X.flatten().argsort() lin_regplot(X[sort_idx], y[sort_idx], tree)

 

On page no: 19

The term "unit step function” is not entirely correct here since the class labels are {1, -1} and not {1, 0}. The best term to use would be just be “piecewise-defined function” in this case.

Page 28: If the link 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' doesn't work/open then please try below link

https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data

Errata Type: Grammar, Page number 133 PDF:

It is:

for i inrange(len(eigen_vals))]

Should be:

for i in range(len(eigen_vals))]

Errata Type: Grammar, Page number 142 PDF:

It is:

S_B += n * (mean_vec - mean_overall).dot(

Should be:

...S_B += n * (mean_vec - mean_overall).dot(

Errata Type: Grammar, Page number 145 PDF:

It is:

plt.scatter(X_train_lda[y_train==l, 0]*(-1)
... X_train_lda[y_train==l, 1]*(-1)

Should be:

plt.scatter(X_train_lda[y_train==l, 0]*(-1),
... X_train_lda[y_train==l, 1]*(-1),

Errata Type : Typo | Page No. 289

This: "plt.show()w" 

Should be: " plt.show()"

Errata Type : Typo | Page No. 51

This:

Later in Chapter 5, Compressing Data via Dimensionality Reduction, 
we will discuss the best practices around model evaluation in more detail

Should be:

Later in Chapter 6, Learning Best Practices for Model 
Evaluation and Hyperparameter Tuning

Errata Type: Code | Page 241

This:
text = re.sub('[\W]+', ' ', text.lower()) + \
'.join(emoticons).replace('-', '')

Should be:
text = re.sub('[\W]+', ' ', text.lower()) + \
''.join(emoticons).replace('-', '')

Errata type: Grammar l Page no: 43

This:data can be discarded after updating the model if storage space in an issue

Should be: data can be discarded after updating the model if storage space is an issue


Errata Typo: Chapter 3 Page 72 
This: We can then we use the parameter C to control the width of the margin and therefore tune the bias-variance trade-off as illustrated in the 
following figure 
Should be: We can then use the parameter C to control the width of the margin and therefore tune the bias-variance trade-off as illustrated in the 
following figure

Errata Type : Graphics| Page : 147 
The 2 figures should be swapped round. The top figure is actually for the test data. The bottom figure is actually for the training data.

Errata Type: Code | Chapter: 7 | Page: 229

It is:  max_depth=None,

Should be:  max_depth=1,

Errata type: Typo | Page number: 378

This:

Assuming that we named our modified ~ a new MLP with 10 hidden layers.

Should be:

Assuming that we named our modified ~ a new MLP with 10 hidden units.

Errata Type: Technical|Page Number: 58

 It is:

 phi(z)=0.5

Should be:

 phi(0) = 0.5

Errata Type: Typo | Page 116

Current content :
The first sentence of the third prose block of text reads: 
"Since we the fit the LogisticRegression object"

It should be:
"Since we fit the LogisticRegression object"
