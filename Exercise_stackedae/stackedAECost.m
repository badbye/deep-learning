function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
n = numel(stack);
% z = cell(n, 1);
% a = cell(n, 1);

%%% feedforward
% z{1} = stack{1}.w * data + repmat(stack{1}.b, 1, M);
% a{1} = sigmoid(z{1});
% for d = 2:n
%    z{d} = stack{d}.w * a{d-1} + repmat(stack{d}.b, 1, M);
%    a{d} = sigmoid(z{d});
% end


z2 = stack{1}.w * data + repmat(stack{1}.b, 1, M);
a2 = sigmoid(z2);
z3 = stack{2}.w * a2+ repmat(stack{2}.b, 1, M);
a3 = sigmoid(z3);

pre = softmaxTheta * a3;
pre = bsxfun(@minus, pre, max(pre, [], 1));
pre = exp(pre);
pre = bsxfun(@rdivide, pre, sum(pre));

% To implement fine tuning, we need to consider all three layers as a 
% single model. Implement stackedAECost.m to return the cost and gradient 
% of the model. The cost function should be as defined as the log likelihood 
% and a gradient decay term. The gradient should be computed using back-
% propagation as discussed earlier. The predictions should consist of the 
% activations of the output layer of the softmax model.To help you check 
% that your implementation is correct, you should also check your gradients 
% on a synthetic small dataset. We have implemented checkStackedAECost.m to 
% help you check your gradients. If this checks passes, you will have 
% implemented fine-tuning correctly.

% Note: When adding the weight decay term to the cost, you should regularize 
% only the softmax weights (do not regularize the weights that compute the 
% hidden layer activations).

cost = -1/M * sum(sum(groundTruth .* log(pre))) + ...
        lambda/2 * (sum(sum(softmaxTheta .^ 2)));

% phi = cell(n+1, 1);
% phi{n+1} = -softmaxTheta' * (groundTruth .- pre) .* log(pre) .* (1 .- log(pre));
% softmaxThetaGrad = phi{n+1} * a{n}' / M + lambda * softmaxThetaGrad; ???

phi4 = -(groundTruth - pre);
phi3 = [softmaxTheta' * phi4] .* a3 .* (1 - a3);
phi2 = stack{n}.w' * phi3 .* a2 .* (1 - a2);

softmaxThetaGrad = phi4 * a3' ./ M  + lambda*softmaxTheta;

stackgrad{n}.w = phi3 * a2' ./ M;
stackgrad{n}.b = mean(phi3, 2);
stackgrad{n-1}.w = phi2 * data' ./ M;
stackgrad{n-1}.b = mean(phi2, 2);
% -------------------------------------------------------------------------
%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
