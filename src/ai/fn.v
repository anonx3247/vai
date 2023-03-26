module ai

import math

type Func = fn (f64) f64

fn relu(x f64) f64 {
	if x < 0 {
		return 0
	} else if x >= 0 && x <= 1 {
		return x
	} else {
		return 1
	}
}

fn (f Func) str() string {
	match f {
		sigmoid { return 'sigmoid' }
		tanh { return 'tanh' }
		relu { return 'relu' }
		else { return 'unknown' }
	}
}

fn sigmoid(x f64) f64 {
	return 1 / (math.exp(x) + math.exp(-x))
}

fn tanh(x f64) f64 {
	return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
}

// Loss functions

fn logloss(pred Tensor, real Tensor) Tensor {
	assert pred.len == real.len, 'pred and real dimensions do not match ${pred.len} != ${real.len}'
	x := pred.data
	y := real.data
	mut arr := []f64{len: pred.len, init: -(y[index] * math.log(x[index]) +
		(1 - y[index]) * math.log(1 - x[index]))}
	return array_to_tensor(arr)
}

fn square_mean(pred Tensor, real Tensor) Tensor {
	return random_tensor(pred.len)
}
