1. Linear regression machine learning algorithm falls under supervised learning

2. In Linear regression, we will learn a model to predict a label's value given a set of features that describe the label

3. Learning a model means, identifying a “hypothesis function” as described below that optimally fits the given data points and can optimally generate the label value for any new set of features
Hypothesis function: y = mx + b So, the hypothesis function depends on the values of “m” and “b”

4. Our goal is to find the optimal values for “m” and “b” (these are also called as weights or parameters), such that the predicted value of the label will be very close to the actual value. To know whether the predicted value of the label is close the actual value or not, we use a function called “Cost Function”. Generally, Mean Square Error function is used as a cost function. The lower the value of the cost function, it indicates that the predicted value of the label is close to the actual value. So, our goal is to minimize the value of the cost function

5. Generally, to decrease the cost function, we increase the number of features in our model. As we keep on increasing the features in model, model starts fitting the training data set well and cost function value starts decreasing.

6. But, with increase in number of features; our equations become a higher order polynomial equation; and it leads to overfitting of the data. 

7. Overfitting of data is bad: In an overfitted model the training error becomes almost zero resulting into saying that model is working perfectly on training data set. But does that model work perfectly on data sets other than training data set like real outside world data? Generally, it is seen that an overfitted model performs worse on the testing data set, and it is also observed that overfitted model perform worse on additional new test data set as well.

8. To fix the problem of overfitting, we need to balance two things:
    
    o	How well function/model fits data
    o	Magnitude of coefficients
   
   So, Total Cost Function = Measure of fit of model + Measure of magnitude of coefficient
    
    o	If Measure of fit of the model is a small value that means model is well fit to the data.    
    o	If Measure of magnitude of coefficient is a small value that means model is not overfit.

9. So, for Ridge regression we use the below Cost function.

    Cost Function: MSE + λ*||W||²
    
    Note:
    1. MSE - Mean Square Error
    2. W - Coefficients
    3. λ here is a tuning parameter to balance the fit of data and magnitude of coefficients.

10. The process of finding λ value is kind of brute force. But with smart guesses and experience, iterations to guess λ value can be reduced.
