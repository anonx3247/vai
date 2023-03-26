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

fn logloss(pred []f64, real []f64) []f64 {
	assert pred.len == real.len, 'pred and real dimensions do not match ${pred.len} != ${real.len}'
	return []f64{len: pred.len, init: -(real[index] * math.log(pred[index]) +
		(1 - real[index]) * math.log(1 - pred[index]))}
}

fn square_mean(pred []f64, real []f64) []f64 {
	return random_array(pred.len)
}
