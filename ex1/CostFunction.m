function J = costFunctionJ (X,y,theta)
m = size(X,1);
predictions = X*theta;
examples
sqrErrors = (predictions-y).^2;

J = 1/(2*m) * sum(sqrErrors);

