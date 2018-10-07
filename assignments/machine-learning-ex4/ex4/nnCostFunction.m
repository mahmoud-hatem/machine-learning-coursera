function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


# feedforward to calculate h
#printf("size of X %dx%d\n", size(X));
a1 = [ones(m,1) X];	# a1 = m * (s1 + 1) , add bias
#printf("size of a1 %dx%d\n", size(a1));
#Theta1 = s2 * (s1 + 1) 
z2 = a1 * Theta1';	# z2 = m * s2
a2 = sigmoid(z2);	# a2 = m * s2

a2 = [ones(m,1) a2];	# a2 = m * (s2 + 1) , add bias
#Theta1 = s3 * (s2 + 1) 
z3 = a2 * Theta2';	# z3 = m * s3
a3 = sigmoid(z3);	# a3 = m * s3

h = a3;			# h = m * s3

# calculate cost J
y_vec = zeros(m, num_labels);
for i= 1:num_labels
    y_vec(:,i) = (y == i);
end

J = (1 / m) * sum(sum( (-y_vec .* log(h)) - ((1 - y_vec) .* log(1 - h)) ));
# add regularization term
theta_unrolled = [Theta1(:, 2:end)(:); Theta2(:, 2:end)(:)];
J = J + ( (lambda / (2 * m)) *  sum(theta_unrolled.^2));

# implement backpropagation

delta_3 = zeros(m, num_labels);		# m * s3
delta_2 = zeros(m, hidden_layer_size);	# m * (s2)

big_delta_2 = zeros(num_labels, hidden_layer_size + 1);
big_delta_1 = zeros(hidden_layer_size,  input_layer_size + 1);


delta_3 = h - y_vec;
delta_2 = (delta_3 * Theta2)(:, 2:end) .* sigmoidGradient(z2);

big_delta_2 = big_delta_2 + (delta_3' * a2);
big_delta_1 = big_delta_1 + (delta_2' * a1);

# using for-loop
#for i = 1:m
#    delta_3(i,:) = h(i, :) - y_vec(i,:);
#
#    #(delta_3(i,:) * Theta2) = 1 * (s2 + 1)
#    delta_2(i,:) = (delta_3(i,:) * Theta2)(2:end) .* sigmoidGradient(z2(i,:));
#    
#    big_delta_2 = big_delta_2 + (delta_3(i,:)' * a2(i,:));
#    big_delta_1 = big_delta_1 + (delta_2(i,:)' * a1(i,:));
#end

Theta2_grad = big_delta_2 ./ m;
Theta1_grad = big_delta_1 ./ m;

# add regularization part
Theta2_grad(:, 2:end) += ((lambda/m) * Theta2(:, 2:end));
Theta1_grad(:, 2:end) += ((lambda/m) * Theta1(:, 2:end));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
