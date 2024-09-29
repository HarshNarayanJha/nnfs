# Okay so we have got some -ve values somewhere in the input or in the hidden layers
# We can't just process -ves.
# 1. ReLU? Clips them to 0, lost info
# 2. Abs, more info is lost!
#
# Solution, exponentiation [exp(x)], it ALWAYS gives +ve output, even for -ve numbers, so we do have a scale with info now!

# SoftMax Activation: Input -> Exp -> Normalize -> Output
# SoftMax Activation: Input -> Softmax -> Output
#
# But exponentiation comes with a cost, for large values it just explodes, with exp(1000) being just overflowed too much
# We can solve this by taking the largest input value and making it 0 (subtrating that value from each value)

import numpy as np

layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026],
]

# Exponentiation
exp_values = np.exp(layer_outputs)
print(exp_values)

# Normalize them
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values, np.sum(norm_values, axis=1))
