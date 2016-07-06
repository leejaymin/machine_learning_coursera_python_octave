# Assignment Linear Regression #

----------

## Programming Excerise 1: Linear Regression ##

The file structure is as bellow.

ex1.m - Octave/MATLAB script that steps you through the exercise
ex1 multi.m - Octave/MATLAB script for the later parts of the exercise
ex1data1.txt - Dataset for linear regression with one variable
ex1data2.txt - Dataset for linear regression with multiple variables
submit.m - Submission script that sends your solutions to our servers
[*] warmUpExercise.m - Simple example function in Octave/MATLAB
[*] plotData.m - Function to display the dataset
[*] computeCost.m - Function to compute the cost of linear regression
[*] gradientDescent.m - Function to run gradient descent
[†] computeCostMulti.m - Cost function for multiple variables
[†] gradientDescentMulti.m - Gradient descent for multiple variables
[†] featureNormalize.m - Function to normalize features
[†] normalEqn.m - Function to compute the normal equations

`*` indicates files you will need to complete
`†` indicates optional exercises


## warmUpExercise.m ##

This is wam up excerise.
Just make indentity matric 5x5.

open the files and write `A=eye(5)`.
Then submit files to the server.

I am done for firt problem according to instruction of course.
```r
>> warmUpExercise()

ans =

     1     0     0     0     0
     0     1     0     0     0
     0     0     1     0     0
     0     0     0     1     0
     0     0     0     0     1

>> submit()
== Submitting solutions | Linear Regression with Multiple Variables...
Login (email address): leejaymin@gmail.com
Token: QthyIC5bQxQOPOXf
== 
==                                   Part Name |     Score | Feedback
==                                   --------- |     ----- | --------
==                            Warm-up Exercise |  10 /  10 | Nice work!
==           Computing Cost (for One Variable) |   0 /  40 | 
==         Gradient Descent (for One Variable) |   0 /  50 | 
==                       Feature Normalization |   0 /   0 | 
==     Computing Cost (for Multiple Variables) |   0 /   0 | 
==   Gradient Descent (for Multiple Variables) |   0 /   0 | 
==                            Normal Equations |   0 /   0 | 
==                                   --------------------------------
==                                             |  10 / 100 | 
== 
```


## Linear regression with one variable ##

Suppose you are the CEO of restaurant franchise and are considering different cities for opning a new branch.

You'd like to use this data to help you select which city to expand to next.

The file `ex1data1.txt` contains the dataset for our linear regression problem.
The first column is the population of a city and the second column is the profit of a food truck in that city.
A negative value for profit indicates loss.

```r
6.1101,17.592
5.5277,9.1302
8.5186,13.662
7.0032,11.854
5.8598,6.8233
8.3829,11.886
7.4764,4.3483
8.5781,12
```

### 2.1 Plotting the Data ###

```python
plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s'); % Set the y−axis label
xlabel('Population of City in 10,000s'); % Set the x−axis label
```
![](http://i.imgur.com/wcMMfLo.jpg)


### 2.2 Gradient Descent ###

In this part, you will fit the linear regression parameters $\theta$ to our dataset using `gradient descent`.

**2.2.1 Update Equations**

The objective of linear regression is to minimize the cost function:


$$J(\theta)=\frac{ 1 }{ 2m } \sum_{ i=1 }^{ m }{ ( h_{\theta}(x^{(i)}) - y^{(i)})^{2}}$$


where the hypothesis $h_{\theta}(x)$ given by the linear model

hθ(x) = θT x = θ0 + θ1x1

Recall that the parameters of your model are the θj values. 
These are the values you will adjust to minimize cost J(θ). One way to do this is to use the batch gradient descent algorithm. 
In batch gradient descent, each iteration performs the update

$$\theta_{j} := \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}{(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}}$$

(simultaneously update $\theta_{j}$ for all j).

With each step of gradient descent, your parameters \theta_{j} come closer to the optimal values that will achieve the lowest cost $J(\theta)$.

**2.2.2 Implementation**

In `ex1.m`, we have already set up the data for linear regression.
In the following lines, we add another dimension to our data to accomodate the $\theta{0}$ intercept term.
We also initialize the initial parameters to 0 and the learning rate `alpha` to `0.01`.

```r
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;
```

**2.2.3 Computing the cost $J(\theta)$**
As you perform gradient descent to learn minimize the cost function J(\theta),
it is helpful to monitor the convergence by computing the cost.
In this section, you will implement a function to calculate $J(\theta)$ so you can check the convergence of your gradient descent implementation.

Your next task is to complete the code in the file computeCost.m, which
is a function that computes J(θ). As you are doing this, remember that the
variables X and y are not scalar values, but matrices whose rows represent
the examples from the training set.
Once you have completed the function, the next step in ex1.m will run
computeCost once using θ initialized to zeros, and you will see the cost
printed to the screen.

You should expect to see a cost of `32.07`.

solution is as below:
```matlab
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = 1/(2*m) * (X * theta - y)' * (X * theta - y); % vectorized form


% =========================================================================
end
```

`apostrophe` means that a matrix will be transposed. 
Finally, above equation is equal to $J(\theta)=\frac{ 1 }{ 2m } \sum_{ i=1 }^{ m }{ ( h_{\theta}(x^{(i)}) - y^{(i)})^{2}}$


If you don't remember vertorized form, below lecture can make you to be getting better understanding.
For full contents, go back to the `linear algebra lecuture`.
![](https://camo.githubusercontent.com/c5b4e5f98214141ad3268fbc459bfccc67009b65/687474703a2f2f692e696d6775722e636f6d2f54455a484e57462e6a7067)

```r
%% =================== Part 3: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
computeCost(X, y, theta)
```

The result
```r
>> computeCost(X, y, theta)

ans =

   32.0727
```

Submit()
```r
>> submit()
== Submitting solutions | Linear Regression with Multiple Variables...
Use token from last successful submission (leejaymin@gmail.com)? (Y/n): Y
== 
==                                   Part Name |     Score | Feedback
==                                   --------- |     ----- | --------
==                            Warm-up Exercise |  10 /  10 | Nice work!
==           Computing Cost (for One Variable) |  40 /  40 | Nice work!
==         Gradient Descent (for One Variable) |   0 /  50 | 
==                       Feature Normalization |   0 /   0 | 
==     Computing Cost (for Multiple Variables) |   0 /   0 | 
==   Gradient Descent (for Multiple Variables) |   0 /   0 | 
==                            Normal Equations |   0 /   0 | 
==                                   --------------------------------
==                                             |  50 / 100 | 
```


**2.2.4 Gradient descent**

Next, you will implement gradient descent in the file `gradientDescent.m`.
The loop structure has been written for you, and you only need to supply the updates to `θ` within each iteration.

As you program, make sure you understand what you are trying to optimize and what is being updated. 
Keep in mind that the cost `J(θ)` is parameterized by the vector `θ`, not X and y. 
That is, we minimize the value of `J(θ)` by changing the values of the vector `θ`, not by changing `X` or `y`. Refer to the equations in this handout and to the video lectures if you are uncertain.

A good way to verify that gradient descent is working correctly is to look at the value of `J(θ)` and check that it is decreasing with each step. 
The starter code for `gradientDescent.m` calls `computeCost` on every iteration and prints the cost. 
Assuming you have implemented gradient descent and computeCost correctly, your value of `J(θ)` should never increase, and should converge to a steady value by the end of the algorithm.

After you are finished, `ex1.m` will use your final parameters to plot the linear fit. 
The result should look something like Figure 2:

![](http://i.imgur.com/yRYbTJ4.png)


Your final values for `θ` will also be used to make predictions on profits in areas of `35,000` and `70,000` people. 
Note the way that the following lines in `ex1.m` uses matrix multiplication, rather than explicit summation or looping, to calculate the predictions. 
This is an example of code vectorization in
Octave/MATLAB.
```r
predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;
```


```matlab
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    Update = 0;
    for i=1:m,
        Update = Update + alpha/m * (theta' * X(i,:)' - y(i)) *  X(i,:)';
    end

    theta = theta - Update;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
```

what is meaning of `:`.
It is that selected all elements.

So, two eqautions are same.
$$\theta_{j} := \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m}{(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}}$$

```r
    for i=1:m,
        Update = Update + alpha/m * (theta' * X(i,:)' - y(i)) *  X(i,:)';
    end
```


without loop, we can implement above function using vectorization.

That is as below:
```matlab
% or a better way
H_theta = X * theta;
grad = (1/m).* x' * (H_theta - y);
theta = theta - alpha .* grad;
```


submit()
```r
>> submit()
== Submitting solutions | Linear Regression with Multiple Variables...
Use token from last successful submission (leejaymin@gmail.com)? (Y/n): Y
== 
==                                   Part Name |     Score | Feedback
==                                   --------- |     ----- | --------
==                            Warm-up Exercise |  10 /  10 | Nice work!
==           Computing Cost (for One Variable) |  40 /  40 | Nice work!
==         Gradient Descent (for One Variable) |  50 /  50 | Nice work!
==                       Feature Normalization |   0 /   0 | 
==     Computing Cost (for Multiple Variables) |   0 /   0 | 
==   Gradient Descent (for Multiple Variables) |   0 /   0 | 
==                            Normal Equations |   0 /   0 | 
==                                   --------------------------------
==                                             | 100 / 100 | 
== 
>> 
```
