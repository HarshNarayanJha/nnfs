# Okay so we have got some -ve values somewhere in the input or in the hidden layers
# We can't just process -ves.
# 1. ReLU? Clips them to 0, lost info
# 2. Abs, more info is lost!
#
# Solution, exponentiation [exp(x)], it ALWAYS gives +ve output, even for -ve numbers, so we do have a scale with info now!

import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

# Exponentiation
exp_values = np.exp(layer_outputs)
print(exp_values)

# Normalize them
norm_values = exp_values / np.sum(exp_values)
print(norm_values, sum(norm_values))
