# Introduction-to-machine-learning-session5-6
هدف:‌ آشنایی با پایه های یادگیری ماشین و کتابخانه Scikit-Learn

آشنایی با اصول تئوری یادگیری آماری و رگرسیون
یادگیری ریاضیات و مفاهیم پایه ای رگرسیون خطی
حل مثال عملی با استفاده از کتابخانه Scikit-Learn با استفاده از رگرسیون خطی


جلسه ششم
هدف‌: آشنایی با رگرسیون غیرخطی (Polynomial) و Logistic Regression

آشنایی با اصول تئوری رگرسیون غیرخطی
حل مثال عملی با استفاده از Polynomial Regression و مقایسه آن با رگرسیون خطی
آشنایی با روش های Classification
آشنایی با اصول تئوری Logistic Regression
حل مثال عملی با استفاده از Logistic Regression

www.hermesai.ir

<img src="https://i.postimg.cc/prCJq27D/scikit-learn.png"  width="300p"/>

# scikit-learn (Sklearn)
One of the most prominent Python libraries for machine learning:

* Contains many state-of-the-art machine learning algorithms
* Builds on numpy (fast), implements advanced techniques
* Wide range of evaluation measures and techniques
* Offers [comprehensive documentation](https://scikit-learn.org/stable/index.html) about each algorithm
* Widely used, and a wealth of [tutorials](http://scikit-learn.org/stable/user_guide.html) and code snippets are available 
* Works well with numpy, scipy, pandas, matplotlib,...

 ## Data import
Multiple options:

* A few toy datasets are included in `sklearn.datasets`
* Import 1000s of datasets via `sklearn.datasets.fetch_openml`
* You can import data files (CSV) with `pandas` or `numpy`

We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:

* Avg. Session Length: Average session of in-store style advice sessions.
* Time on App: Average time spent on App in minutes
* Time on Website: Average time spent on Website in minutes
* Length of Membership: How many years the customer has been a member.

![newplot](https://user-images.githubusercontent.com/100142624/181494622-c48cd14f-ffcd-419a-b369-784aa64e7fd7.png)
![newplot (1)](https://user-images.githubusercontent.com/100142624/181494689-52fcbf9e-9f89-45d1-a37c-d7c4bf88df9c.png)



## Building models
All scikitlearn estimators follow the same interface

   *  **fit():**    Fit/model the training data
   
   
   * **predict():**  Make predictions
    
    
   *  **score():**  Predict and compare to true

## Training and Testing Data

Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
* Set a variable X equal to the numerical features of the customers and
* a variable y equal to the "Yearly Amount Spent" column.
### Training and testing data
To evaluate our model, we need to test it on unseen data.  
`train_test_split`: splits data randomly.


## Fitting a model
We'll build a Linear Regression model


### Predicting Test Data
Now that we have fit our model, let's evaluate its performance by predicting off the test values!


![newplot (2)](https://user-images.githubusercontent.com/100142624/181494925-881a6d99-1dc1-43a1-a7c4-ac24138480b3.png)
### Evaluating the model
Feeding all test examples to the model yields all predictions
<img src="https://i.postimg.cc/CxSzwGRL/metrics.png" width="400" />



## Logistic Regression

In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

* **Daily Time Spent on Site:** consumer time on site in minutes
* **Age:** cutomer age in years
* **Area Income:** Avg. Income of geographical area of consumer
* **Daily Internet Usage:** Avg. minutes a day consumer is on the internet
* **Ad Topic Line:** Headline of the advertisement
* **City:** City of consumer
* **Male:** Whether or not consumer was male
* **Country:** Country of consumer
* **Timestamp:** Time at which consumer clicked on Ad or closed window
* **Clicked on Ad:** 0 or 1 indicated clicking on Ad


![newplot (3)](https://user-images.githubusercontent.com/100142624/181495035-99e093e9-b33c-460d-b14f-11e4978dccb0.png)
![newplot (4)](https://user-images.githubusercontent.com/100142624/181495057-1b8a303b-f92b-4978-88e0-a5285f8e5a40.png)
![newplot (5)](https://user-images.githubusercontent.com/100142624/181495077-b35804dd-f22a-4805-bce2-dae265300ada.png)
![newplot (5)](https://user-images.githubusercontent.com/100142624/181495105-f5d6b2c7-20fc-46be-a167-d45a121a2022.png)


## confusion matrix

A confusion matrix is a table that is used to define the performance of a classification algorithm.
<img src="https://i.postimg.cc/bvG8Pr9w/confusion2.png" width ="700"/>


* **True Positive:**

Interpretation: You predicted positive and it’s true.

You predicted that a woman is pregnant and she actually is.

* **True Negative:**

Interpretation: You predicted negative and it’s true.

You predicted that a man is not pregnant and he actually is not.

* **False Positive:** (Type 1 Error)

Interpretation: You predicted positive and it’s false.

You predicted that a man is pregnant but he actually is not.

* **False Negative:** (Type 2 Error)

Interpretation: You predicted negative and it’s false.

You predicted that a woman is not pregnant but she actually is.
##  How to Calculate Confusion Matrix for a 2-class classification problem?

### Recall
<img src="https://i.postimg.cc/23Nz6DTB/recall.png" width ="300"/>
The above equation can be explained by saying, from all the positive classes, how many we predicted correctly.

Recall should be high as possible.
### Precision
<img src="https://i.postimg.cc/Pr1X7Crb/precision.png" width ="300"/>
The above equation can be explained by saying, from all the classes we have predicted as positive, how many are actually positive.

Precision should be high as possible.


It is difficult to compare two models with low precision and high recall or vice versa. So to make them comparable, we use F-Score. F-score helps to measure Recall and Precision at the same time.

### F-measure
<img src="https://i.postimg.cc/dQ7YTLbR/fmeasure.png" width ="400"/>


## Learning Curves

A plot of the training, validation score with respect to the size of the training set is known as a Learning curve.

<img src="https://i.postimg.cc/kGsmnbVL/learning-curve.png"/>  
If we plotted the error scores for each training size, we’d get two learning curves looking similarly to these:
<img src="https://i.postimg.cc/GmV1hr23/learning-curve2.png"/>


**Learning curves give us an opportunity to diagnose bias and variance in supervised learning models.**
<img src="https://i.postimg.cc/BvRT4SGf/bias2.png"/>  
------------------------------------------------------------------------------------

<img src="https://i.postimg.cc/52RvS4yQ/variance.png"/>


### Feature Scaling

Many machine learning algorithms perform better when numerical input variables are scaled to a standard range.

* **Normalization** is the process of scaling data into a range of [0, 1]. It's more useful and common for regression tasks.
* **Standardization** is the process of scaling data so that they have a mean value of 0 and a standard deviation of 1. It's more useful and common for classification tasks.

<img src = "https://i.postimg.cc/j5vg2QX6/scaling2.png" />


## <span style="color:red">ROC curve</span>

An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.

This curve plots two parameters:

TPR & FPR

<img src="https://i.postimg.cc/vBPBcx0B/tpr-fpr.png" width="700" />

An ROC curve plots TPR vs. FPR at different classification thresholds.

Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.

The following figure shows a typical ROC curve.

<img src="https://i.postimg.cc/qvf7hTx8/curve.png" width="700" />

To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient.

Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC.

## <span style="color:red">AUC</span>

AUC stands for "Area under the ROC Curve."

That is, AUC measures the entire two-dimensional area underneath the entire ROC curve

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.


Were is the optimal point?
<img src="https://i.postimg.cc/tgqpWCZV/ROC.png" width="500"/>

Which curve is better?

<img src="https://i.postimg.cc/Bngtdz2H/AUC.png" width="500"/>

