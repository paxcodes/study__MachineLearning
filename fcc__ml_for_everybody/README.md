# Linear Regression

Model to calculate the "line of best fit". "Line of best fit" can be described using the
formula: `y = b0 + b1x` where `b0` is the y-intercept when x=0 and `b1` is the slope
(coefficient)  

`residual/error` - absolute value of the difference between the predicted value and the actual value

`simple linear regression` - we have one feature or x-value: `y = b0 + b1x`

`multiple linear regression` - we have multiple features or x-values: `y = b0 + b1 x1 + b2 x2 + ... + bn xn`

## Assumptions

These assumptions affect the data points or the samples:

1. Linearity - Does my data follow a linear pattern? Does the relationship between the features and the target follow a linear pattern? Does y-decreases as x-increases / x-decreases?
2. Independence - Are the data points independent of each other? Do the data points rely on each other? Does the data point affect other data points?

These assumptions affect the residuals or errors:

When you plot the residuals or errors:

1. Normality - Are the residuals normally distributed?
2. Homoscedasticity - Is the spread of the residuals the same across the data points?

Note that with very large samples, you can dispense with the normality assumption.

> we do not have to do anything if we are analyzing a so-called “large” sample — which typically means 120 observations or more. In other words, whenever you have more than 120 observations in your data, you could dispense with the normality assumption altogether. The reason is that we can invoke the Central Limit Theorem for large samples. But that is a lesson for another day. (source: <https://medium.com/@christerthrane/the-normality-assumption-in-linear-regression-analysis-and-why-you-most-often-can-dispense-with-5cedbedb1cf4>)
