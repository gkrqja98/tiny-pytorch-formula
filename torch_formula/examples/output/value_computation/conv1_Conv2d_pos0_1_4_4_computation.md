# Detailed Computation for Conv2d Layer 'conv1'

## Position: (0, 1, 4, 4)

### Computing output at position: (batch=0, out_channel=1, y=4, x=4)

#### Filter Weights (Shape: (1, 3, 3))

**Channel 0**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.2000 | 0.5000 | -0.3000 | 
| [1] | -0.4000 | 0.6000 | 0.1000 | 
| [2] | 0.7000 | -0.2000 | 0.4000 | 

#### Input Receptive Field (Shape: (3, 3, 1))

**Channel 0**

| Index | [0] | 
| --- | --- | 
| [0] | 1.0000 | 
| [1] | 1.0000 | 
| [2] | 1.0000 | 

**Channel 1**

| Index | [0] | 
| --- | --- | 
| [0] | 1.0000 | 
| [1] | 1.0000 | 
| [2] | 0.8429 | 

**Channel 2**

| Index | [0] | 
| --- | --- | 
| [0] | 1.0000 | 
| [1] | 0.8429 | 
| [2] | 0.9143 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(1.0000 \times 0.2000) + (1.0000 \times 0.5000) + (1.0000 \times -0.3000) + (1.0000 \times -0.4000) + (1.0000 \times 0.6000) + (0.8429 \times 0.1000) + (1.0000 \times 0.7000) + (0.8429 \times -0.2000) + (0.9143 \times 0.4000) + -0.2000$

### Computation Result

Calculated value: 1.381429

Actual output value: 1.381429

