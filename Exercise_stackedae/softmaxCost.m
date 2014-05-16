function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);
m = size(labels, 1);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));

cost = -1/m * sum(sum(groundTruth .* log(M))) + lambda/2 * sum(sum(theta .^ 2));
% [value location] = max(M);
% $$\nabla_{\theta_j} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}{ \left[ x^{(i)} ( 1\{ y^{(i)} = j\}  - p(y^{(i)} = j | x^{(i)}; \theta) ) \right]  } + \lambda \theta_j$$
for i = 1:numClasses
    % label = full(sparse(labels == i, 1:numCases, 1));
    loss = repmat([labels == i]' - M(i, :), inputSize, 1);
	thetagrad(i, :) =  -mean(data .* loss, 2)' + lambda * theta(i, :);
end


%%% thetagrad = -mean(data .* (groundTruth - M), 2) + lambda * theta;








% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

