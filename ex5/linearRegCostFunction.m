function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

n = size(theta, 1);
h = X*theta;
squareD = (h - y).^2;

theta2 = theta(2:end, 1);

J = 0.5/m*sum(squareD) + 0.5*lambda/m*sum(theta2.^2);

temp = X'*(X*theta - y);

grad =  1/m*temp + lambda/m*theta;

grad(1) = 1/m*temp(1);


% =========================================================================

grad = grad(:);

end
