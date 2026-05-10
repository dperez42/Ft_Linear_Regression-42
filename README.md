# FT Linear Regression (a 42 school Project)

The goal of this project is to introduce us to the basic concepts of ML.

We will have to implement a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

*The implementation of this project will be done in Python.*

## Table of Contents
- [Usage](#usage) 
- [Concepts needed for the project](#concepts-needed-for-the-project) 🧠
	- [Linear Regression](#linear-regression) 📈
	- [Gradient Descent](#gradient-descent) 📉
	- [Loss Function](#loss-function) 🧮
	- [Feature Scaling](#feature-scaling) 📏
	- [Wrapping it up](#wrapping-it-up) 🎁
- [Resources](#resources) 📖

# Usage

## Trainning

**Usage: python train.py data_file.csv <flags>**

Choose Standarization method flags:
-sm (default): Min Max Feature Scaling.
-ss : Standarization Feature Scaling.

Choose Loss function flags:
-mae (default): Mean Absolute Error.
-mse : Mean Squared Error.

Choose Objective flags:
-dd (default= 0.0000001): Define delta loss between epochs.
-de : Define number of epochs.

Choose Plotting flags:
-po : Plot raw data.
-ps : Plot standarize data.
-pl : Plot loss.

Choose Logs flags:
-err : To view logs while trainning.

python .\train.py data.csv  -mse -dd -sm -po -ps -pl -err

## Make a prediction

python .\predict.py model.csv

You will prompted to enter kms.

# Concepts needed for the project

## Linear Regression

Simple Linear regression is a statistical technique used to find the relationship between variables. In an ML context, linear regression finds the relationship between a feature and a label.

The formula for a linear regression is:

$$
y = θ_{0} + θ_{1}x
$$

Where:

- $y$ is the target variable. The output
- $x$ is the feature. The input.
- $θ_{0}$ is the bias. Bias is the same concept as the y-intercept in the algebraic equation for a line. Bias is a parameter of the model and is calculated during training.
- $θ_{1}$ is the weight. Weight is the same concept as the slope $m$ in the algebraic equation for a line. Weight is a parameter of the model and is calculated during training.

It's literally a line equation, like $y = mx + b$ in school.

In our case, the target variable is the price of a car and the feature is the kms.

> For example, if we find out that $price = -0.5 * mileage$, it means that for each unit of mileage, the price of the car decreases by 0.5.

## Loss Function

In ML, a loss function Loss is a numerical metric that describes how wrong a model's predictions are. Loss measures the distance between the model's predictions and the actual labels. The goal of training a model is to minimize the loss, reducing it to its lowest possible value.

It measures the **difference between the predicted values and the actual values. Thus, all methods for calculating loss remove the sign.**.

The two most common methods to remove the sign are the following:

- Take the absolute value of the difference between the actual value and the prediction.
- Square the difference between the actual value and the prediction.

### Types of loss

In linear regression, there are five main types of loss.

- $L_{1} loss$: The sum of the absolute values of the difference between the predicted values and the actual values.
- Mean absolute error $(MAE)$: The average of $L_{1}$ losses across a set of N examples.
- $L_{2} loss$: The sum of the squared difference between the predicted values and the actual values.
- Mean squared error $(MSE)$: The average of $L_{2}$ losses across a set of N examples.
- Root mean squared error $(RMSE)$: The square root of the mean squared error (MSE).

The functional difference between L1 loss and L2 loss (or between MAE/RMSE and MSE) is squaring. When the difference between the prediction and label is large, squaring makes the loss even larger. When the difference is small (less than 1), squaring makes the loss even smaller.

Loss metrics like MAE and RMSE may be preferable to L2 loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error using the same scale as the model's predicted value.

When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE.

Note the relationship between the model and the data:

- MSE. The model is closer to the outliers but further away from most of the other data points.
- MAE. The model is further away from the outliers but closer to most of the other data points.


The loss function used in our linear regression is the **Mean Absolute Error** (MAE).

$$
MAE = \frac{1}{m}  \sum_{i=1}^{m} | \hat{y_{i}} - y_{i} |
$$

Where:

- $m$ is the number of samples
- $\hat{y}$ is the predicted value
- $y$ is the actual value

To make it simpler, **it means we are going to estimate the price for a mileage, subtract it to the actual price in the data we have, do this for all the data points, and then divide the sum by the number of data points**.

For example, if we have 3 data points:

- $y_{1} = 1000$ and $\hat{y}_{1} = 900$
- $y_{2} = 2000$ and $\hat{y}_{2} = 1900$
- $y_{3} = 3000$ and $\hat{y}_{3} = 3100$

The MAE will be:

$$
MAE = \frac{|1000 - 900| + |2000 - 1900| + |3000 - 3100|}{3} = 100
$$



Here, the MAE is 100, which means that our model is off by 100 on average.

Other loss function used in our linear regression is the **Mean Squared Error** (MSE).

$$
MSE = \frac{1}{m}  \sum_{i=1}^{m} ( \hat{y_{i}} - y_{i} )^2
$$

Where:

- $m$ is the number of samples
- $\hat{y}$ is the predicted value
- $y$ is the actual value

Using the same data points of above the MSE will be:

$$
MSE = \frac{(1000 - 900)^2 + (2000 - 1900)^2 + (3000 - 3100)^2}{3} = 10000
$$

Here, the MSE is 10000, which means that our model is off by 10000 on average.

> Unless your data points are perfectly aligned, the MAE will never be 0, it is totally normal to have a loss.

Now that we know how to measure the precision of our model, we need to find the best $θ_{0}$ and $θ_{1}$ that minimize this loss function.

## Gradient Descent

Gradient descent is an optimization algorithm that consists of finding the minimum of a function by iteratively getting closer to it.

In our case, the function we want to minimize is the loss function.

The algorithm works as follows:

1. Initialize the $θ_{0}$ and $θ_{1}$ to 0
2. Calculate the gradient of the loss function, that is to say "*what do $θ_{0}$ and $θ_{1}$ miss to be optimal?*"
3. Update the $θ_{0}$ and $θ_{1}$ in the opposite direction of the gradient (subtract it)
4. Repeat steps 2 and 3 until the loss function converges (i.e. stagnates)

The formula to update the $θ_{0}$ and $θ_{1}$ when using **MAE** loss function is:

$$
θ_{0} = θ_{0} - α \frac{1}{m} \sum_{i=1}^{m}(\hat{y}\_{i} - y\_{i})
$$

$$
θ_{1} = θ_{1} - α \frac{1}{m} \sum_{i=1}^{m}((\hat{y}\_{i} - y\_{i}) * x\_{i})
$$

When using **MSE** loss function is:

$$
θ_{0} = θ_{0} - α \frac{1}{m} \sum_{i=1}^{m}(\hat{y}\_{i} - y\_{i})*2
$$

$$
θ_{1} = θ_{1} - α \frac{1}{m} \sum_{i=1}^{m}((\hat{y}\_{i} - y\_{i}) *2 x\_{i})
$$

Where:

- $α$ is the learning rate
- $m$ is the number of samples
- $\hat{y}$ is the predicted value
- $y$ is the actual value
- $x$ is the feature

> The learning rate is a hyperparameter that controls how much we update the $θ_{0}$ and $θ_{1}$ at each iteration.
>
> If it's too high, we might overshoot the minimum, if it's too low, we might take too long to converge.

Why do we use these formulas to update the $θ_{0}$ and $θ_{1}$?

- **$θ_{0}$'s update is straightforward, it's just the average of the errors**: if your line is ≈ 100 above the actual values, you just need to lower it by 100.

- **$θ_{1}$'s update is a bit more complex**, given that it's a coefficient. We need to **correct its offset, but also its slope** (how inclined it is).

> You might have noticed that the two update formulas I provided are literally the ones in the subject:
>
> For MAE:
>
> $tmpθ_{0} = θ_{0} - learningRate * \frac{1}{m} \sum_{i=0}^{m - 1} (estimatePrice(kms[i]) - price[i])$
> $tmpθ_{1} = θ_{1} - learningRate * \frac{1}{m} \sum_{i=0}^{m - 1} (estimatePrice(kms[i]) - price[i]) * kms[i]$
>
> For MSE:
>
> $tmpθ_{0} = θ_{0} - learningRate * \frac{1}{m} \sum_{i=0}^{m - 1} (estimatePrice(kms[i]) - price[i]) * 2$
> $tmpθ_{1} = θ_{1} - learningRate * \frac{1}{m} \sum_{i=0}^{m - 1} (estimatePrice(kms[i]) - price[i]) * 2 * kms[i]$

## Hyperparameters

**Learning rate**

Learning rate is a floating point number you set that influences how quickly the model converges. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges, but instead bounces around the weights and bias that minimize the loss. The goal is to pick a learning rate that's not too high nor too low so that the model converges quickly.

The learning rate determines the magnitude of the changes to make to the weights and bias during each step of the gradient descent process. The model multiplies the gradient by the learning rate to determine the model's parameters (weight and bias values) for the next iteration. In the third step of gradient descent, the "small amount" to move in the direction of negative slope refers to the learning rate.

The difference between the old model parameters and the new model parameters is proportional to the slope of the loss function. For example, if the slope is large, the model takes a large step. If small, it takes a small step. For example, if the gradient's magnitude is 2.5 and the learning rate is 0.01, then the model will change the parameter by 0.025.

A learning rate that's too small can take too many iterations to converge.

A learning rate that's too large never converges because each iteration either causes the loss to bounce around or continually increase.

The ideal learning rate helps the model to converge within a reasonable number of iterations.

**Epochs**

During training, an epoch means that the model has processed every example in the training set once.
Training typically requires many epochs. That is, the system needs to process every example in the training set multiple times.

The number of epochs is a hyperparameter you set before the model begins training. In many cases, you'll need to experiment with how many epochs it takes for the model to converge. In general, more epochs produces a better model, but also takes more time to train.

## Feature Scaling

If we implement this code as is, we might run into a problem: the mileage is in the thousands, while the price is in the tens of thousands.

This means that the $θ_{1}$ will be updated more for the mileage than for the price, which is not what we want.

> Anyways, the code would crash because $θ_{1}$ would be way too high.

To avoid this, we need to scale the features.

In our case, we can choose between **standardization** method and **min-max scaling**.

**standardization**

$$
x_{scaled} = \frac{x - μ}{σ}
$$

Where:

- $x$ is the feature
- $μ$ is the mean of the feature
- $σ$ is the standard deviation of the feature

> The mean is used to center the data around 0
>
> The standard deviation is used to scale the data, so it has a variance of 1 (i.e., the data points are equally spread on the x-axis)

For example, if we have the following mileages:

- $mileage = [1000, 2000, 3000]$
- $μ = 2000$
- $σ = 816.5$

The scaled mileage will be:

$$
mileage_{scaled} = \left[ \frac{1000 - 2000}{816.5}, \frac{2000 - 2000}{816.5}, \frac{3000 - 2000}{816.5} \right] = [-1.22, 0, 1.22]
$$

**min-max scaling**

Pending

## Wrapping it up

Now that we know all the concepts needed for the project, we can summarize the steps to implement the linear regression:

1. Load the data
2. Scale the feature (the mileage)
3. Initialize $θ_{0}$ and $θ_{1}$ to 0
4. Make a naive prediction of $price = θ_{0} + θ_{1} * mileage$ for every data point
5. Calculate the average error
6. Update $θ_{0}$ and $θ_{1}$ accordingly
7. Repeat steps 4 to 6 until the loss converges
8. Save $θ_{0}$ and $θ_{1}$ to a file

# Resources
- [📖 Crash Course Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression)

- [📖 Normalization and Standardization](https://www.geeksforgeeks.org/machine-learning/feature-engineering-scaling-normalization-and-standardization/)

- [📖 How Neural Networks Learn using Gradient Descent](https://bhatnagar91.medium.com/how-neural-networks-learn-using-gradient-descent-f48c2e4079a6)

- [📺 Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)

- [📖 Linear Regression using Gradient Descent](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931)

- [📖 Linear Regression Model using Gradient Descent algorithm](https://dilipkumar.medium.com/linear-regression-model-using-gradient-descent-algorithm-50267f55c4ac)

- [💬 Multiple regression - how to calculate the predicted value after feature normalization?](https://stats.stackexchange.com/a/207752)

- [📺 Linear Regression and Partial Differentiation!](https://www.youtube.com/watch?v=StHyJm5xcjs)

- [📖 14 Loss functions you can use for Regression](https://medium.com/@mlblogging.k/14-loss-functions-you-can-use-for-regression-b24db8dff987)

- **Thanks to [leogaudin](https://github.com/leogaudin) for the explanation of the coefficient and bias update formulas.**