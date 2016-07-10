# 2. Regularized logistic regression #

----------

Microchip 제작에 관한 Quality Assureance (QA)를 예측하는 작업을 수행 한다.

QA동안에 microchip은 많은 test를 겪게 된다.

chips manager라고 가정하고 이제 2개의 test 결과를가지고 해당 chips을 통과 시킬지 말지를 결정하는 것을 생각해 보자.

이번에는 결국 아래의 파일만 구현하면 된다.

- [*] costFunctionReg.m - Regularized Logistic Regression Cost


## 2.1 Visualizing the data ##

이전과 같이 일단 visualization 부터 수행 한다.
`ex2_reg`를 실행하면 기본적으로 아래와 같이 그래프를 그려 준다.
![](http://i.imgur.com/iLHH3C7.png)

해당 분포를 보면 linear dicision boundary로는 결정을 할 수 없는 것을 알 수 있다.

## 2.2 Feature mapping ##

`mapFeature.m`에 의해서 featrue의 수를 증가 시킨다.

![](http://i.imgur.com/RM4oNd0.png)

6 제곱 까지 증가시킨 다항식을 통해서 정확도 높은 dicision boundary를 생성 할 수 있다.

mapping 함수를 통해서 two features vector는 28-dimensional vector로 변경 된다.

이러한 feature mapping은 좀더 설명 가능한 classifier를 생성하게 도와 준다. 하지만 overfitting에 대해서는 민감하게 반응하게 된다.

## 2.3 Cost Function and Gradient ##

regularized logistic regression을 완성하기 위해서 `cost function`과 `gradient function`을 `costFunctionReg.m`에다가 구현 한다.

**Regularized Cost Function**
$$ J(\theta) = \frac{1}{m}\sum_{i=1}^{m}\big[-y^{(i)}\, log\,( h_\theta\,(x^{(i)}))-(1-y^{(i)})\,log\,(1-h_\theta(x^{(i)}))\big] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{j}^{2} $$

**Vectorized Cost Function**
$$ J(\theta) = \frac{1}{m}\big((\,log\,(g(X\theta))^Ty+(\,log\,(1-g(X\theta))^T(1-y)\big) + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{j}^{2} $$

**Partial derivative**
$$ \frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m}\sum_{i=1}^{m} ( h_\theta (x^{(i)})-y^{(i)})x^{(i)}_{j} + \frac{\lambda}{m}\theta_{j} $$

**Vectorized**
$$ \frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m} X^T(g(X\theta)-y) + \frac{\lambda}{m}\theta_{j} $$

$\theta_0$는 regularization 하지 않는다. $j$는 1부터 시작하게 된다.

모든 $\theta$가 0으로 초기화 되었을 때의 cost 값은 `0.693`이다.

**costFunctionReg.m**
```octave
function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

H_theta = sigmoid(X * theta);

% Cost
J =(1/m) * sum(-y .* log(H_theta) - (1 - y) .* log(1 - H_theta)) + ...
   (lambda/(2*m)) * norm(theta([2:end]))^2;

G = (lambda/m) .* theta;
G(1) = 0; % extra term for gradient

% Gradient
grad = ((1/m) .* X' * (H_theta - y)) + G;

% =============================================================

end
```

### 2.3.1 Learning parameters using `fminunc` ###

이전과 같이 `ex2_reg.m`에서 fminunc 함수를 이용해서 $\theta$를 최적화 한다.


## 2.4 Plotting the decision boundary ##

`plotDecisionBoundary.m`에 의해서 non-linear decision boundary를 그리게 된다.

![](http://i.imgur.com/7eLvYNa.png)

그리고 최종적인 train accuracy는 다음과 같다.
```bash
Train Accuracy: 83.050847
```

