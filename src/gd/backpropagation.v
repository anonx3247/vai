module gradient_descent

import ai
import arrays
import util as u

/*
Implementation of the backpropagation algorithm as described here:
https://en.wikipedia.org/wiki/Backpropagation#u.Matrix_multiplication

This algorithm works in the following way:

1. We calculate the partial gradient at layer l denoted delta(l)
2. This is done using the following recursive formula:

delta(l-1) = f_l' * W_(l+1)^T . ... . f_(L-1)' * W_L^T . f_L' grad(C, f_L)


- where f_i is the activation function at layer i
- W_k is the weight matrix of coefficients (i,j) where W_k(i,j) = the weight multiplying the input from neuron j in layer k-1 as output to neuron i in layer k
- ^T is the transpose operation
- C is the cost function

3. Once the deltas have been calculated, we can determine dC/dW_k:

dC/dW_k =delta(k) * f_(k-1)^T


4. Finally given a learning rate eta, we update each weight following the formula:

W_k(i,j) += -eta * dC/dW_k(i,j)


Note: here we have the Cost function evaluated at x comparing to the expected result y:

C(y, f_L(W_L * f_(L-1) (W_(L-1) ... f_1(W_1 * x)))
*/
