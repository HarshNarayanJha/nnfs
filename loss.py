# We need to find the loss of the training in order to improve the model
# Why not success but loss, because we need to correct those only.
#
## Categorical Cross-Entropy: -log(predicted target probability value)

import math
softmax_ouptut = [0.7, 0.1, 0.2]

# One-Hot Vector
# label = 0 (means the desired ouput) and classes are 3
# so the vector has a 1 at index 0 and 0 otherwise
target_output = [1, 0, 0]

# Categorical Cross-Entropy (target_output[1] and [2] are just 0, so loss is just -log(softmax_output[label]))
loss = -(math.log(softmax_ouptut[0]) * target_output[0] +
         math.log(softmax_ouptut[1]) * target_output[1] +
         math.log(softmax_ouptut[1]) * target_output[2])

print(loss)
# OR
loss = -math.log(softmax_ouptut[0]) # 0.35667494393873245

# Let's say the confidence was lower
print(-math.log(0.5))               # 0.6931471805599453

# The loss is now much more, that's what it is
# less confidence = more loss
