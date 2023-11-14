1. For each of parts (a) through (d), indicate whether we would generally
   expect the performance of a flexible statistical learning method to be
   better or worse than an inflexible method. Justify your answer.
   (a) The sample size n is extremely large, and the number of predictors p is small.
   (b) The number of predictors p is extremely large, and the number
   of observations n is small.
   (c) The relationship between the predictors and response is highly
   non-linear.
   (d) The variance of the error terms, i.e. σ2 = Var(ϵ), is extremely
   high.

> a: better, because it can fit the data better
> b: worse, because it can overfit the data
> c: better, because it can fit the data better
> d: worse, because it can overfit the data

2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested
   in inference or prediction. Finally, provide n and p.
   (a) We collect a set of data on the top 500 firms in the US. For each
   firm we record profit, number of employees, industry and the
   CEO salary. We are interested in understanding which factors
   affect CEO salary.

> n=500 p=4
> Inference
> Regression

(b) We are considering launching a new product and wish to know
whether it will be a success or a failure. We collect data on 20
similar products that were previously launched. For each product we have recorded whether it was a success or failure,
price
charged for the product, marketing budget, competition price,
and ten other variables.

> n=20 p=14
> Prediction
> Classification

(c) We are interested in predicting the % change in the USD/Euro
exchange rate in relation to the weekly changes in the world
stock markets. Hence we collect weekly data for all of 2012. For
each week we record the % change in the USD/Euro, the %
change in the US market, the % change in the British market,
and the % change in the German market.

> n=52 p=4
> Prediction
> Regression

5. What are the advantages and disadvantages of a very flexible (versus
   a less flexible) approach for regression or classification? Under what
   circumstances might a more flexible approach be preferred to a less
   flexible approach? When might a less flexible approach be preferred?

> Advantages: can fit the data better, we might use flexible methods when the patterns in the data are highly
> non-linear, or have no obvious pattern
> Disadvantages: can overfit the data, we might use less flexible methods when the patterns in the data are linear or
> have an obvious pattern

6. Describe the differences between a parametric and a non-parametric
   statistical learning approach. What are the advantages of a parametric approach to regression or classification (as
   opposed to a nonparametric approach)?
   What are its disadvantages?

> Parametric: we make an assumption about the functional form, and then fit the model to the data using that form
> Non-parametric: we make no assumption about the functional form, and instead fit the model to the data using a
> flexible approach
> Parametric advantages: simpler, faster, more interpretable
> Parametric disadvantages: can be inaccurate if the functional form is wrong
> Non-parametric advantages: can fit the data better
> Non-parametric disadvantages: can overfit the data

7. The table below provides a training data set containing six observations, three predictors, and one qualitative
   response variable.

| Obs. | X1 | X2 | X3 | Y     |
|------|----|----|----|-------|
| 1    | 0  | 3  | 0  | Red   |
| 2    | 2  | 0  | 0  | Red   |
| 3    | 0  | 1  | 3  | Red   |
| 4    | 0  | 1  | 2  | Green |
| 5    | -1 | 0  | 1  | Green |
| 6    | 1  | 1  | 1  | Red   |

Suppose we wish to use this data set to make a prediction for Y when X1 = X2 = X3 = 0 using K-nearest neighbors.
(a) Compute the Euclidean distance between each observation and the test point, X1 = X2 = X3 = 0.

| Obs. | X1 | X2 | X3 | Y     | Distance to 0 0 0 |
|------|----|----|----|-------|-------------------|
| 1    | 0  | 3  | 0  | Red   | 3                 |
| 2    | 2  | 0  | 0  | Red   | 2                 |
| 3    | 0  | 1  | 3  | Red   | 3.162             |
| 4    | 0  | 1  | 2  | Green | 2.236             |
| 5    | -1 | 0  | 1  | Green | 1.414             |
| 6    | 1  | 1  | 1  | Red   | 1.732             |

(b) What is our prediction with K = 1? Why?

> Green, because the closest point is green (Obs 5)

(c) What is our prediction with K = 3? Why?

> Red, because the closest 3 points are red (Obs 5: Green, 6: Red, 2: Red); Green: 33.33%, Red: 66.66%

(d) If the Bayes decision boundary in this problem is highly nonlinear, then would we expect the best value for K to be
large or small? Why?

> Small, because we want to fit the data better. Being non-linear means that the data is not linearly separable, so we
> need to fit the data better.
