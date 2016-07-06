# Programming Exercise 2: Logistic Regression and Regularized Logistic Regression #

----------
## Introduction ##
여기선 `logistic regression`을 구현하고 서로다른 두개의 datasets에 적용해 본다.

작업해야할 파일들은 아래와 같다.

- `ex2.m` - Octave/MATLAB script that steps you through - the exercise
- `ex2 reg.m` - Octave/MATLAB script for the later parts of the exercise
- `ex2data1.txt` - Training set for the first half of the exercise
- `ex2data2.txt` - Training set for the second half of the exercise
- `submit.m` - Submission script that sends your solutions to our servers
- `mapFeature.m` - Function to generate polynomial features
- `plotDecisionBoundary.m` - Function to plot classifier’s decision boundary

- `[*]` plotData.m - Function to plot 2D classification data
- `[*]` sigmoid.m - Sigmoid Function
- `[*]` costFunction.m - Logistic Regression Cost Function
- `[*]` predict.m - Logistic Regression Prediction Function
- `[*]` costFunctionReg.m - Regularized Logistic Regression Cost

`*`가 들어간 것들이 구현해야될 파일 이다.

`ex2.m`과 `ex2_reg.m`의 파일이 전체적인 exercise를 다루고 있는 것들이다. 이 두 파일은 dataset을 set up하고 문제를 해결하기 위해서 우리가 작성해야할 function들을 호출하는 순서로 구성 되어 있다.

직접적으으로 `ex2.m`과 `ex2_reg.m`을 수정할 필요는 없다. 그저 각각의 다른파일에 저장된 `function`의 내용을 채우면 된다.

## 1. Logistic Regression ##

학생들이 대학에 입할 할 수 있는지 아닌지에 대한 예측을 가능 하도록 하는 Logistic regression classifier를 구현 한다.

과거 지원자의 두 과목 시험 점수와 그에 따른 admission결과를 training set으로 하여 model을 생성하게 된다.

모델로 만약 어떤 사용자의 시험 성정 2개가 주어진다면 이 사용자가 과거 대비 입학을 할 수 있을지에 대한 probability를 구하게 된다.

### 1.1 Visualizing the data ###

데이터 형태 파악을 위해서 일단 plot을 그려 보는것이 효과 적이다.

plotData function을 호출해서 plot을 그리게 된다.

당연히 plotData.m 파일의 함수를 구현해야 그래프가 그려 진다.

**ex2.m**
```octave
#파일 읽기
data = load('ex2data1.txt')

data = 

   34.62366   78.02469    0.00000
   30.28671   43.89500    0.00000
   35.84741   72.90220    0.00000
   60.18260   86.30855    1.00000
   79.03274   75.34438    1.00000
   45.08328   56.31637    0.00000
   61.10666   96.51143    1.00000
   ....

size(data)

ans =

   100     3

# 총 100개의 데이터가 있다.
# 1-2 열이 과목 점수고, 3번째 열이 admission 여부를 나타낸다. 각각을 X와 Y로 끄집어 낸다.
X = data(:,[1,2])
y = data(:,3)

# 두개를 그래프 그리기 위해서 인자로 넘겨 준다.
plotData(X,y)
```

**plotData.m**
```octave
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find Indices of Positive and Negtive Example
pos = find(y == 1);
neg = find(y == 0);

% Plot Example
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'yellow', 'MarkerSize', 7);


% =========================================================================

```

![](http://i.imgur.com/nzAyHQ0.png)

### 1.2 Implementation ###

#### 1.2.1 Warmup exercise: sigmoid function ####

그냥 몸풀기 정도로 logistic regression에서 사용하는 hypothesis는 다음과 같다.

$$ h_{\theta} = g(\theta^{T} x) $$

여기서 g는 sigmoid function이다. sigmoid는 아래와 같다.

$$ g(z) = \frac{1}{1+e^{-z}} $$
위 sigmoid를 `sigmoid.m`에다가 구현 한다.

```octave
function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1.0 ./ (1.0 + exp(-z));

% =============================================================
end
```
코드에서 `./`는 엘리먼트 나눗셈이다.
즉, 두개의 배열이 있을때 각각의 엘리먼트 끼리 나누는 것이다.
z인자가 matrix로 들어올 수 있기 때문에 이렇게 해줘야 한다.

**sigmoid test**
아래와 같이 잘 구현 되었는지 확인해보면
큰 positive value면 1이, 큰 negative value면 0이
0이면 0.5로 잘 나오는것을 알 수 있다.
```bash
octave:7> sigmoid(100000000)
ans =  1
octave:8> sigmoid(-100000000)
ans = 0
octave:9> sigmoid(0)
ans =  0.50000
```

그리고 matrix형태로 z 값을 넣어보면, 아래와 같이 각각에 대해서 sigmoid가 잘 적용되는 것을 알 수 있다.
```bash
octave:10> sigmoid(data)
ans =

   1.00000   1.00000   0.50000
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.50000
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.73106
   1.00000   1.00000   0.50000
```

제출
```bash
octave:1> submit()
== Submitting solutions | Logistic Regression...
Use token from last successful submission (leejaymin@gmail.com)? (Y/n): 
== 
==                                   Part Name |     Score | Feedback
==                                   --------- |     ----- | --------
==                            Sigmoid Function |   5 /   5 | Nice work!
==                    Logistic Regression Cost |   0 /  30 | 
==                Logistic Regression Gradient |   0 /  30 | 
==                                     Predict |   0 /   5 | 
==        Regularized Logistic Regression Cost |   0 /  15 | 
==    Regularized Logistic Regression Gradient |   0 /  15 | 
==                                   --------------------------------
==                                             |   5 / 100 | 

```


#### 1.2.2 Cost function and gradient ####

logistic regression의 cost function을 설계해 본다.
`costFunction.m`을 구현 하면 된다.


$$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m}{-y^{(i)} log(h_{\theta}(x^{(i)})) - (1-y^{(i)}) log(1-h_{\theta}(x^{(i)})) } $$

**vectorized cost function**
$$ J(\theta) = \frac{1}{m}\big((\,log\,(g(X\theta))^Ty+(\,log\,(1-g(X\theta))^T(1-y)\big) $$

이 `Cost function`을 gradient descent 알고리즘을 적용 하면 아래와 같다.

Partial derivative
$$ \frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m}\sum_{i=1}^{m} ( h_\theta (x^{(i)})-y^{(i)})x^{(i)}_{j} $$

Vectorized partial derivative
$$ \frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m} X^T(g(X\theta)-y) $$



gradient descent formula 자체는 linear regression 때와 동일하다. 하지만 $h_{\theta}x$의 정의가 다른 것이다.

이제 초기 $\theta$의 값을 0,0,0 으로해서 계산을 수행해 보면
cost 값은 `0.693`이 나와야 한다.

**costFunction.m**

```octave
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

H_theta = sigmoid(X * theta);

J = (1.0/m) * sum(-y .* log(H_theta) - (1.0 - y) .* log(1.0 - H_theta));

grad = (1.0/m) .* X' * (H_theta - y);


% =============================================================

end

```
X matrix에 theta를 모두 곱한것이 z가 된다.
즉, $\theta^{T}x$인 향태인 것이다.
theta_0,1,2 순으로 저장이 되어 있다.

결국
```bash
octave:31> size(X)
ans =

   100     3

octave:32> size(initial_theta)
ans =

   3   1
```
[100,3] * [3,1] 을 수행해서 [100,1] 짜리 matrix를 생성 한다.
당연히 초기 theta들은 모두 0이므로 sigmoid(0)은 0.5이므로 H_thetha는 0으로 채워져 있다.

```bash
-y .* log(H_theta) - (1.0 - y) .* log(1.0 - H_theta)
```
의 값은 0.69315이고 이것이 100개 있다. 이대로 쭉 계산하면 

1/100 * (0.69315) * 100 이므로 결과는 `0.69315`가 나온다.


두 번쨰로 `gradient descent formula`구현한 코드는 
```octave
grad = (1.0/m) .* X' * (H_theta - y);
```
이다. 위와 같이 matrix 연산을 통해서 모든 training data와 모든 `wegiht vector` ($\theta$)에 대해서 한번에 처리하게 된다.

그리고 이것에 대한 초기 gradient 조정 값은 아래와 같다.
```bash
Gradient at initial theta (zeros): 
 -0.100000 
 -12.009217 
 -11.262842 
```

**제출**
```bash
==                                   Part Name |     Score | Feedback
==                                   --------- |     ----- | --------
==                            Sigmoid Function |   5 /   5 | Nice work!
==                    Logistic Regression Cost |  30 /  30 | Nice work!
```


#### 1.2.3 Learning parameters using fminunc ####

Linear regression에서는 gradient descent algorithm을 직접 구현 했었다. 하지만 여기서는 Octave에서 제공하는 `fminunc`라는 함수를 이용할 것이다.

**fminunc**
- initial values of the parameters we are trying to optimize.
- A function that, when given the training set and a particular $\theta$, computes the logistic regression cost and gradient with respect to $\theta$ for the dataset (X,y)

이것을 위해서는 어떤 parameters를 넣으면 될지는 아래와 같다.
```octave
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```
`GradObj`를 option `on`으로 설정 했다.
gradient를 수행해서 최적화 하겠다는 의미이다.
`MaxIter`의 option은 `400`으로 설정 했다. 400번 이후로는 그냥 terminates 된다.

fminunc는 최적의 costfunction 값이 나올 때가지 400번 반복 한다. 그다음 최적의 weight vector가 저장된 theta와 그때의 cost를 반환 하게 된다.

재밌는 것은 지난번과 달리 `loop`를 만들지도 않았고 `learning rate`도 따로 설정하지 않았다.
`fminunc`를 사용할때 설정할것은 cost와 gradient를 계산해줄 함수만 넣어주면 된다.

결국 아래의 과정이 수행된 것이다.
![](http://i.imgur.com/c4QcDOE.png)

`ex2.m`을 실행하면 최적의 $\theta$ 값과 `cost`값으로 아래와 같이 출력 한다.
```bash
Program paused. Press enter to continue.

Cost at theta found by fminunc: 0.203498

theta: 
 -25.161272 
 0.206233 
 0.201470 
```

그리고 이미 작성된 코드 `plotDecisionBoundary.m`에 의해서 아래와 같이 `Decision Boundary`가 그려 진다.
![](http://i.imgur.com/qYy6FWi.png)

#### 1.2.4 Evaluating logisitic regression ####

이제 학습된 파라메터들을 이용해서 (wegiht vector, $\theta$) 어떤 학생의 시험점수 2개가 있다면 이 학생이 admission을 받을지 안받을지를 예측해 볼 수 있다.

예로
Exam 1 score: 45
Exam 2 score: 85
위와 같을때 admission probability는 `0.776`이다. 코드는 아래와 같다.

```octave
prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);
```


이것을 위해서 `predict.m`파일을 구현 한다.

**predict.m**
```octave
function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters.
%               You should set p to a vector of 0's and 1's
%

p = sigmoid(X * theta);

idx_1 = find(p >= 0.5);
idx_0 = find(p < 0.5);

p(idx_1) = ones(size(idx_1));
p(idx_0) = zeros(size(idx_0));

% =========================================================================
end
```

이제 작성된 `predict`함수를 이용해서 training data에 대한 prediction 정확도를 보자.

```octave
% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
```
정확도가 높이 나왔다. 이러한 검증 방식은 사실 문제가 있다.
학습 데이터로 모델을 평가하면 상당히 성능이 좋게 나오기 때문이다. overffting이 되면 될 수록 성능은 더 좋기 때문이다.
substitution error라고도 부른다.
```bash
Train Accuracy: 89.000000
```

여기까지 했을 때의 점수 이다.

```bash
==                                   Part Name |     Score | Feedback
==                                   --------- |     ----- | --------
==                            Sigmoid Function |   5 /   5 | Nice work!
==                    Logistic Regression Cost |  30 /  30 | Nice work!
==                Logistic Regression Gradient |  30 /  30 | Nice work!
==                                     Predict |   5 /   5 | Nice wor
```



