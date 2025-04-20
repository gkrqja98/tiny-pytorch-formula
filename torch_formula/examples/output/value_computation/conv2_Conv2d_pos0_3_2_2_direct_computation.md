# Detailed Computation for Conv2d Layer 'conv2' (Direct Calculation)

## Position: (0, 3, 2, 2)

### Computing output at position: (batch=0, out_channel=3, y=2, x=2)

#### Filter Weights (Shape: (2, 3, 3))

**Channel 0**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.1146 | 0.1580 | -0.0716 | 
| [1] | -0.0861 | 0.0379 | 0.0591 | 
| [2] | 0.0640 | 0.1835 | 0.0529 | 

**Channel 1**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.0044 | 0.0237 | -0.0345 | 
| [1] | 0.0657 | -0.1527 | -0.0430 | 
| [2] | 0.2193 | -0.2283 | -0.1697 | 

#### Input Receptive Field (Shape: (3, 3, 2))

**Channel 0**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.8643 | 1.2243 | 
| [1] | 2.2229 | 1.3929 | 
| [2] | 1.9200 | 1.1829 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 2.0186 | 1.4814 | 
| [1] | 1.9471 | 1.3814 | 
| [2] | 1.8729 | 1.3857 | 

**Channel 2**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.7957 | 1.0443 | 
| [1] | 2.0714 | 1.3814 | 
| [2] | 1.8914 | 1.3686 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(1.8643 \times 0.1146) + (2.2229 \times 0.1580) + (1.9200 \times -0.0716) + (2.0186 \times -0.0861) + (1.9471 \times 0.0379) + (1.8729 \times 0.0591) + (1.7957 \times 0.0640) + (2.0714 \times 0.1835) + (1.8914 \times 0.0529) + (1.2243 \times 0.0044) + (1.3929 \times 0.0237) + (1.1829 \times -0.0345) + (1.4814 \times 0.0657) + (1.3814 \times -0.1527) + (1.3857 \times -0.0430) + (1.0443 \times 0.2193) + (1.3814 \times -0.2283) + (1.3686 \times -0.1697) + -0.1285$

### Computation Result

Calculated value: 0.410543

Actual output value: 0.410543

