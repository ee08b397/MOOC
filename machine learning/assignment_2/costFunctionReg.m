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

[J, grad] = costFunction(theta, X, y);

size_theat = size(theta);

J = J + sum(theta(2:size_theat,1) .* theta(2:size_theat,1)) * lambda / (2 * m);

for j = 2:size_theat,
    grad(j) = grad(j) + lambda * theta(j) / m;
end;
%sumforJ = 0;
%for i = 1:m,
%    sumforJ = y(i) * log(sigmoid(X(i,:) * theta)) + (1 - y(i)) * ...
%        log(1 - sigmoid(X(i,:) * theta)) + sumforJ;
%end;
%J = -sumforJ / m + sum(theta .* theta) * lambda / (2 * m);
%
%sumforG = 0;
%for i = 1:m,
%    sumforG = (sigmoid(X(i,:) * theta) - y(i)) * X(i,1) + sumforG;
%end;
%grad(1) = sumforG / m;
%
%
%for j = 2:size(theta),
%    sumforG = 0;
%    for i = 1:m,
%        sumforG = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j) + sumforG;
%    end;
%    grad(j) = sumforG / m + lambda * theta(j) / m;
%end;


% =============================================================

end
