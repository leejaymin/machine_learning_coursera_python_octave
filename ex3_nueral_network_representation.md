# Programming Exercise 3: Multi-class classification and Neural Networks #

----------

## Introduction ##

one-vs-all logistic regression과 neural networks를 구현하고 이를 통해서 hand-written digits를 인식해 볼 것이다.

## Files included in this exercise ##

ex3.m - Octave/MATLAB script that steps you through part 1
ex3 nn.m - Octave/MATLAB script that steps you through part 2
ex3data1.mat - Training set of hand-written digits
ex3weights.mat - Initial weights for the neural network exercise
submit.m - Submission script that sends your solutions to our servers
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine (similar to fminunc)
sigmoid.m - Sigmoid function
[*] lrCostFunction.m - Logistic regression cost function
[*] oneVsAll.m - Train a one-vs-all multi-class classifier
[*] predictOneVsAll.m - Predict using a one-vs-all multi-class classifier
[*] predict.m - Neural network prediction function


전반적으로 `ex3.m`과 `ex3_nn.m`파일을 가지고 작업을 하게 된다.

## 1. Multi-class Classification ##

우선, 이전에 개발한 `Logistic regression`을 확장해서 `one-vs-all classification`을 구현 하게 된다.

### 1.1 Dataset ###

`ex3data1.mat`에 저장되어 있다.
이 파일에는 5000개의 handwritten digit들이 training example로써 5000개 저장되어 있다.
`.mat` format은 native Ooctave/MATLAB matrixv 포멧을 의미한다.

정확히 저장 되었다면 결과 값이 matrix에 저장 되게 된다. 

```matlab
% Load saved matrices from file
load('ex3data1.mat');
% The matrices X and y will now be in your Octave environment
```

20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector.
결국 Eample 하나는 single row가 되고 matrix는
5000 by 400 matrix가 된다.
![](http://i.imgur.com/RNkp3PW.png)

호환성을 위해서 0을 10으로 mapping 하고 나머지는 그대로 mapping 한다.

### 1.2 Visualizing the data ###

displayData 코드는 랜덤으로 100개의 image들을 선택해서 보여주게 된다.

![](http://i.imgur.com/01XJ24O.png)


### 1.3 Vectorizing Logistic Regression ###

Training 시간을 줄이기 위해서 vectorization을 반드시 수행 해야 한다.

#### 1.3.1 Vectorizing the cost function ####

vectorization을 이용해서 unregularized cost function을 `lrCostFunction.m`에 다가 구현을 한다.
반드시 fully vectorized version을 만들어야 한다.
이것의 의미는 loop가 없어야 함을 의미한다.

#### 1.3.2 Vectorizing the gradient ####

gradient descent algorithm도 vectorization을 통해서 적용을 하게 된다.

`Cost Function`과 `Gradient`를 모두 적용한 matlab 코드는 아래와 같다.


#### 1.3.3 Vectorizing regularized logistic regression ####

Regularzation 까지 적용한 Vectorization을 생성 한다.
최종 코드는 아래와 같다.

**lrCostFunction.m**
```matlab
function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

H_theta = sigmoid(X * theta);

% Cost
J =(1/m) * sum(-y .* log(H_theta) - (1 - y) .* log(1 - H_theta)) + ...
   (lambda/(2*m)) * norm(theta([2:end]))^2;

G = (lambda/m) .* theta;
G(1) = 0; % extra term for gradient

% Gradient
grad = ((1/m) .* X' * (H_theta - y)) + G;


% =============================================================

grad = grad(:);

end
```

### 1.4 One-vs-all Classification ###

one-vs-all classification을 multiple regularized logistic regression 이용한 방법이다.

작성해야할 코드는 `oneVsAll.m`이다.


**oneVsAll.m**
```matlab
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
%
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

for k = 1:num_labels

    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 500); % iteration could be bigger, e.g. 500

    [theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == k), lambda)), ...
                 initial_theta, options);

    all_theta(k,:) = theta';
end
% =========================================================================
end
```

#### 1.4.1 One-vs-all Prediction ####

주어진 image를 prediction 해보자.
10개의 classifier가 존재하고 이중에서 가장 높은 `probability`를 가지는 것으로 prediction 하게 된다.

작성해야할 코드는 `predictOneVsAll` 함수 이다.
이전에 학습된 #\theta$를 이용해서 prediction을 수행하게 된다.
정확도는 `94.9%`를 기록하게 된다.

**predictOneVsAll.m**
```matlab
function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the
%       max element, for more information see 'help max'. If your examples
%       are in rows, then, you can use max(A, [], 2) to obtain the max
%       for each row.
%

[x, ix] = max(sigmoid(X * all_theta'), [], 2);

p = ix;

% =========================================================================
end
```

단순 logistic regression으로도 아래와 같이 훌륭한 결과를 얻을 수 있다.

```bash
Loading and Visualizing Data ...
Program paused. Press enter to continue.

Training One-vs-All Logistic Regression...
Iteration   457 | Cost: 1.311466e-02
Iteration   500 | Cost: 5.080189e-02
Iteration   500 | Cost: 5.760187e-02
Iteration   500 | Cost: 3.306492e-02
Iteration   500 | Cost: 5.445615e-02
Iteration   500 | Cost: 1.825591e-02
Iteration   500 | Cost: 3.064138e-02
Iteration   500 | Cost: 7.845164e-02
Iteration   500 | Cost: 7.118580e-02
Iteration   500 | Cost: 8.567845e-03
Program paused. Press enter to continue.

Training Set Accuracy: 96.460000
octave:38> 
```

## 2. Neural Networks ##

이전까지는 Logistic regression을 다중으로 사용해서 이미지 숫자를 구별하는 방법을 했었다.

여기서는 handwritten digits을 인식하기 위해서 neural network을 이용하게 된다.

하지만 parameter trained은 직접 하지 않고 이미 학습한 $\theta$값을 이용한다.

여기서는 이미 정의된 $\theta$를 이용해서 feedforward propagation algorithm을 구현 하는데 의미가 있다.

### 2.1 Model representation ###

입력, 히든, 출력 이렇게 3개의 레이어로 구성된 neural network 이다.

입력은 20x20 이미지 이기때문에 400이라고 할 수 있다.
이것에 bias unit 값으로 +1이 추가 된다.

neural network에서 쓰이는 parameter들은 이미 학습 되어져서 `ex3weights.mat`에 저장되어 있다.
히든 레이어는 `25개`의 unit으로 구성되어 있고 output 레이어는 10개의 unit으로 구성되어 있다.

이전에 강의 자료에서 설명되었듯이 $\theta^{j}$의 dimension은 $s_{j+1} \times s_{j}+1$로 정의 된다.
따라서 아래와 같은 matrix를 구성하게 된다. 
```matlab
% Load saved matrices from file
load('ex3weights.mat');
% The matrices Theta1 and Theta2 will now be in your Octave
% environment
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
```

![](http://i.imgur.com/1Jka4YB.png)

**<Neural Network Model>**

### 2.2 Feedforward Propagation and Prediction ###

`predict.m`을 구현 함으로써 feedforward propagaction을 구현하게 된다.

어떤 example $i$에 대해서 $h_{\theta}(x^{(i)})$를 이용해서 digit을 prediction 하게 된다.
one-vs-all classification과 같이 가장큰 확률을 가지는 i class로 prediction 된다.

**predict.m**
```matlab
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m, 1) X]; % add 1

z2 = a1 * Theta1';

a2 = [ones(size(sigmoid(z2), 1), 1) sigmoid(z2)]; % add 1

z3 = a2 * Theta2';
a3 = sigmoid(z3); % H_theta(x)

[x, ix] = max(a3, [], 2);

p = ix;

% =========================================================================
end
```

결과를 확인해보면,
우선 ex3_nn.m은 미리 학습에 의해서 정해진 Theta1과 Theta2를 불러온다.
그런다음 예제 5000개가 저장된 X를 입력해서 prediction 하게 된다.

최종 결과를 참값 y를 통해서 비교하게된다.
정확도는 97.5%가 나오게 된다.

그다음에는 하나하나 이미지를 보여주면서 생성한 모델이 어떻게 prediction 하는지 확인 할 수 있다.

**실행결과**
```bash
Loading and Visualizing Data ...
Program paused. Press enter to continue.

Loading Saved Neural Network Parameters ...

Training Set Accuracy: 97.520000
Program paused. Press enter to continue.

Displaying Example Image

Neural Network Prediction: 8 (digit 8)
Program paused. Press enter to continue.
```
![](http://i.imgur.com/7okS8Lj.png)



