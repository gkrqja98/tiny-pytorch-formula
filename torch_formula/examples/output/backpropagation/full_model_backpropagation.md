# Complete Forward and Backward Analysis of TinyCNN Model

## Input Shape: torch.Size([1, 1, 8, 8])
## Output Shape: torch.Size([1, 2])
## Loss Value: 0.305365

## Layer: conv1 (Conv2d)

- **Input Shape**: torch.Size([1, 1, 8, 8])
- **Output Shape**: torch.Size([1, 2, 8, 8])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Computing output at position: (batch=0, out_channel=0, y=1, x=1)

#### Filter Weights (Shape: (1, 3, 3))

**Channel 0**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.3000 | -0.5000 | 0.2000 | 
| [1] | 0.7000 | 0.4000 | -0.1000 | 
| [2] | -0.3000 | 0.8000 | 0.5000 | 

#### Input Receptive Field (Shape: (3, 3, 1))

**Channel 0**

| Index | [0] | 
| --- | --- | 
| [0] | 0.2000 | 
| [1] | -0.0286 | 
| [2] | 0.3429 | 

**Channel 1**

| Index | [0] | 
| --- | --- | 
| [0] | -0.0286 | 
| [1] | 0.3429 | 
| [2] | 0.1143 | 

**Channel 2**

| Index | [0] | 
| --- | --- | 
| [0] | 0.3429 | 
| [1] | 0.1143 | 
| [2] | 0.4857 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(0.2000 \times 0.3000) + (-0.0286 \times -0.5000) + (0.3429 \times 0.2000) + (-0.0286 \times 0.7000) + (0.3429 \times 0.4000) + (0.1143 \times -0.1000) + (0.3429 \times -0.3000) + (0.1143 \times 0.8000) + (0.4857 \times 0.5000) + 0.1000$

### Computation Result

Calculated value: 0.580000

Actual output value: 0.580000

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

### Backward Computation for Conv2d at Position: (batch=0, out_channel=0, y=1, x=1)

#### Gradient Output Value

-0.003830

#### Filter Weights

#### Filter Weights (Shape: (1, 3, 3))

**Channel 0**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.3000 | -0.5000 | 0.2000 | 
| [1] | 0.7000 | 0.4000 | -0.1000 | 
| [2] | -0.3000 | 0.8000 | 0.5000 | 

#### Gradient Propagation to Input

| Position (y, x, c) | Formula | Gradient Value |
| ----------------- | ------- | -------------- |
| (0, 0, 0) | $(-0.0038 \times 0.3000)$ | -0.001149 |
| (0, 1, 0) | $(-0.0038 \times -0.5000)$ | 0.001915 |
| (0, 2, 0) | $(-0.0038 \times 0.2000)$ | -0.000766 |
| (1, 0, 0) | $(-0.0038 \times 0.7000)$ | -0.002681 |
| (1, 1, 0) | $(-0.0038 \times 0.4000)$ | -0.001532 |
| (1, 2, 0) | $(-0.0038 \times -0.1000)$ | 0.000383 |
| (2, 0, 0) | $(-0.0038 \times -0.3000)$ | 0.001149 |
| (2, 1, 0) | $(-0.0038 \times 0.8000)$ | -0.003064 |
| (2, 2, 0) | $(-0.0038 \times 0.5000)$ | -0.001915 |

#### Weight Gradient Computation

| Weight Position (out_c, in_c, ky, kx) | Formula | Gradient Value |
| ------------------------------------ | ------- | -------------- |
| (0, 0, 0, 0) | $(-0.0038 \times 0.2000)$ | -0.000766 |
| (0, 0, 0, 1) | $(-0.0038 \times -0.0286)$ | 0.000109 |
| (0, 0, 0, 2) | $(-0.0038 \times 0.3429)$ | -0.001313 |
| (0, 0, 1, 0) | $(-0.0038 \times -0.0286)$ | 0.000109 |
| (0, 0, 1, 1) | $(-0.0038 \times 0.3429)$ | -0.001313 |
| (0, 0, 1, 2) | $(-0.0038 \times 0.1143)$ | -0.000438 |
| (0, 0, 2, 0) | $(-0.0038 \times 0.3429)$ | -0.001313 |
| (0, 0, 2, 1) | $(-0.0038 \times 0.1143)$ | -0.000438 |
| (0, 0, 2, 2) | $(-0.0038 \times 0.4857)$ | -0.001860 |

#### Bias Gradient Computation

Bias Gradient for output channel 0: -0.003830

### General Gradient Formulas

#### Input Gradient Formula

$\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

#### Weight Gradient Formula

$\frac{\partial L}{\partial w_{c_{out},c_{in},k_h,k_w}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot x_{n,c_{in},h_{in}+k_h,w_{in}+k_w}$

#### Bias Gradient Formula

$\frac{\partial L}{\partial b_{c_{out}}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}}$



## Layer: pool1 (MaxPool2d)

- **Input Shape**: torch.Size([1, 2, 8, 8])
- **Output Shape**: torch.Size([1, 2, 4, 4])

### Forward Pass

- **General Formula**: $y_{n,c,h_{out},w_{out}} = \max_{k_h,k_w} x_{n,c,h_{in}+k_h,w_{in}+k_w}$

### Computing output at position: (batch=0, channel=0, y=1, x=1)

#### Input Receptive Field (Shape: (2, 2))

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.2314 | 1.4557 | 
| [1] | 1.7514 | 1.8643 | 

Maximum value position in receptive field: (1, 1) with value 1.8643

### General Formula

$y_{n,c,h_{out},w_{out}} = \max_{0 \leq i < k_h, 0 \leq j < k_w} x_{n,c,h_{in}+i,w_{in}+j}$

### Value Substitution

$\max(1.2314, 1.4557, 1.7514, 1.8643)$

### Computation Result

Calculated max value: 1.864286

Actual output value: 1.864286

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \begin{cases} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}} & \text{if } x_{n,c,h_{in},w_{in}} \text{ is max in pool} \\ 0 & \text{otherwise} \end{cases}$

### Backward Computation for MaxPool2d at Position: (batch=0, channel=0, y=1, x=1)

#### Gradient Output Value

-0.003542

#### Input Receptive Field

#### Receptive Field (Shape: (2, 2))

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.2314 | 1.4557 | 
| [1] | 1.7514 | 1.8643 | 

Maximum value position in receptive field: (1, 1) with value 1.864286

#### Gradient Propagation to Input

| Position (y, x) | Formula | Gradient Value |
| -------------- | ------- | -------------- |
| (3, 3) | $-0.0035$ | -0.003542 |

All other positions in the receptive field receive zero gradient.

#### Gradient Map (Receptive Field)

#### Gradient Map (Shape: (2, 2))

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.0000 | 0.0000 | 
| [1] | 0.0000 | -0.0035 | 

### General Gradient Formula

$\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \begin{cases} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}} & \text{if } x_{n,c,h_{in},w_{in}} \text{ is max in pool} \\ 0 & \text{otherwise} \end{cases}$



## Layer: conv2 (Conv2d)

- **Input Shape**: torch.Size([1, 2, 4, 4])
- **Output Shape**: torch.Size([1, 4, 4, 4])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Computing output at position: (batch=0, out_channel=0, y=1, x=1)

#### Filter Weights (Shape: (2, 3, 3))

**Channel 0**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.5000 | 0.1862 | 0.0398 | 
| [1] | -0.0827 | -0.2315 | -0.1452 | 
| [2] | -0.0208 | 0.2278 | -0.0299 | 

**Channel 1**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.3000 | -0.0553 | 0.0391 | 
| [1] | 0.2137 | -0.2156 | 0.0396 | 
| [2] | 0.1689 | 0.0722 | 0.1024 | 

#### Input Receptive Field (Shape: (3, 3, 2))

**Channel 0**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.5800 | 0.3629 | 
| [1] | 1.3000 | 0.9214 | 
| [2] | 1.3686 | 0.9700 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.4657 | 0.9514 | 
| [1] | 1.8643 | 1.2243 | 
| [2] | 2.2229 | 1.3929 | 

**Channel 2**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.2486 | 0.9400 | 
| [1] | 2.0186 | 1.4814 | 
| [2] | 1.9471 | 1.3814 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(0.5800 \times 0.5000) + (1.3000 \times 0.1862) + (1.3686 \times 0.0398) + (1.4657 \times -0.0827) + (1.8643 \times -0.2315) + (2.2229 \times -0.1452) + (1.2486 \times -0.0208) + (2.0186 \times 0.2278) + (1.9471 \times -0.0299) + (0.3629 \times 0.3000) + (0.9214 \times -0.0553) + (0.9700 \times 0.0391) + (0.9514 \times 0.2137) + (1.2243 \times -0.2156) + (1.3929 \times 0.0396) + (0.9400 \times 0.1689) + (1.4814 \times 0.0722) + (1.3814 \times 0.1024) + 0.0500$

### Computation Result

Calculated value: 0.634222

Actual output value: 0.634222

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

### Backward Computation for Conv2d at Position: (batch=0, out_channel=0, y=1, x=1)

#### Gradient Output Value

-0.005189

#### Filter Weights

#### Filter Weights (Shape: (2, 3, 3))

**Channel 0**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.5000 | 0.1862 | 0.0398 | 
| [1] | -0.0827 | -0.2315 | -0.1452 | 
| [2] | -0.0208 | 0.2278 | -0.0299 | 

**Channel 1**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | 0.3000 | -0.0553 | 0.0391 | 
| [1] | 0.2137 | -0.2156 | 0.0396 | 
| [2] | 0.1689 | 0.0722 | 0.1024 | 

#### Gradient Propagation to Input

| Position (y, x, c) | Formula | Gradient Value |
| ----------------- | ------- | -------------- |
| (0, 0, 0) | $(-0.0052 \times 0.5000)$ | -0.002595 |
| (0, 1, 0) | $(-0.0052 \times 0.1862)$ | -0.000966 |
| (0, 2, 0) | $(-0.0052 \times 0.0398)$ | -0.000207 |
| (1, 0, 0) | $(-0.0052 \times -0.0827)$ | 0.000429 |
| (1, 1, 0) | $(-0.0052 \times -0.2315)$ | 0.001201 |
| (1, 2, 0) | $(-0.0052 \times -0.1452)$ | 0.000754 |
| (2, 0, 0) | $(-0.0052 \times -0.0208)$ | 0.000108 |
| (2, 1, 0) | $(-0.0052 \times 0.2278)$ | -0.001182 |
| (2, 2, 0) | $(-0.0052 \times -0.0299)$ | 0.000155 |
| (0, 0, 1) | $(-0.0052 \times 0.3000)$ | -0.001557 |
| (0, 1, 1) | $(-0.0052 \times -0.0553)$ | 0.000287 |
| (0, 2, 1) | $(-0.0052 \times 0.0391)$ | -0.000203 |
| (1, 0, 1) | $(-0.0052 \times 0.2137)$ | -0.001109 |
| (1, 1, 1) | $(-0.0052 \times -0.2156)$ | 0.001119 |
| (1, 2, 1) | $(-0.0052 \times 0.0396)$ | -0.000206 |
| (2, 0, 1) | $(-0.0052 \times 0.1689)$ | -0.000876 |
| (2, 1, 1) | $(-0.0052 \times 0.0722)$ | -0.000374 |
| (2, 2, 1) | $(-0.0052 \times 0.1024)$ | -0.000531 |

#### Weight Gradient Computation

| Weight Position (out_c, in_c, ky, kx) | Formula | Gradient Value |
| ------------------------------------ | ------- | -------------- |
| (0, 0, 0, 0) | $(-0.0052 \times 0.5800)$ | -0.003010 |
| (0, 0, 0, 1) | $(-0.0052 \times 1.3000)$ | -0.006746 |
| (0, 0, 0, 2) | $(-0.0052 \times 1.3686)$ | -0.007102 |
| (0, 0, 1, 0) | $(-0.0052 \times 1.4657)$ | -0.007606 |
| (0, 0, 1, 1) | $(-0.0052 \times 1.8643)$ | -0.009674 |
| (0, 0, 1, 2) | $(-0.0052 \times 2.2229)$ | -0.011535 |
| (0, 0, 2, 0) | $(-0.0052 \times 1.2486)$ | -0.006479 |
| (0, 0, 2, 1) | $(-0.0052 \times 2.0186)$ | -0.010475 |
| (0, 0, 2, 2) | $(-0.0052 \times 1.9471)$ | -0.010104 |
| (0, 1, 0, 0) | $(-0.0052 \times 0.3629)$ | -0.001883 |
| (0, 1, 0, 1) | $(-0.0052 \times 0.9214)$ | -0.004781 |
| (0, 1, 0, 2) | $(-0.0052 \times 0.9700)$ | -0.005034 |
| (0, 1, 1, 0) | $(-0.0052 \times 0.9514)$ | -0.004937 |
| (0, 1, 1, 1) | $(-0.0052 \times 1.2243)$ | -0.006353 |
| (0, 1, 1, 2) | $(-0.0052 \times 1.3929)$ | -0.007228 |
| (0, 1, 2, 0) | $(-0.0052 \times 0.9400)$ | -0.004878 |
| (0, 1, 2, 1) | $(-0.0052 \times 1.4814)$ | -0.007687 |
| (0, 1, 2, 2) | $(-0.0052 \times 1.3814)$ | -0.007169 |

#### Bias Gradient Computation

Bias Gradient for output channel 0: -0.005189

### General Gradient Formulas

#### Input Gradient Formula

$\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

#### Weight Gradient Formula

$\frac{\partial L}{\partial w_{c_{out},c_{in},k_h,k_w}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot x_{n,c_{in},h_{in}+k_h,w_{in}+k_w}$

#### Bias Gradient Formula

$\frac{\partial L}{\partial b_{c_{out}}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}}$



## Layer: pool2 (AvgPool2d)

- **Input Shape**: torch.Size([1, 4, 4, 4])
- **Output Shape**: torch.Size([1, 4, 2, 2])

### Forward Pass

- **General Formula**: $y_{n,c,h_{out},w_{out}} = \frac{1}{k_h \cdot k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}$

### Computing output at position: (batch=0, channel=0, y=1, x=1)

#### Input Receptive Field (Shape: (2, 2))

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 1.7873 | 1.9932 | 
| [1] | 0.9956 | 1.1012 | 

Kernel size: (2, 2)

### General Formula

$y_{n,c,h_{out},w_{out}} = \frac{1}{k_h \cdot k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}$

### Value Substitution

$\frac1{2 \times 2} \times (1.7873 + 1.9932 + 0.9956 + 1.1012)$

### Computation Result

Calculated average value: 1.469322

Actual output value: 1.469322

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \frac{1}{k_h \cdot k_w} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}}$

### Backward Computation for AvgPool2d at Position: (batch=0, channel=0, y=1, x=1)

#### Gradient Output Value

0.007160

Pool Size: 4 (kernel: (2, 2))

Distributed Gradient: 0.001790 (0.007160 / 4)

#### Gradient Propagation to Input

| Position (y, x) | Formula | Gradient Value |
| -------------- | ------- | -------------- |
| (2, 2) | $0.0072 / 4$ | 0.001790 |
| (2, 3) | $0.0072 / 4$ | 0.001790 |
| (3, 2) | $0.0072 / 4$ | 0.001790 |
| (3, 3) | $0.0072 / 4$ | 0.001790 |

#### Gradient Map (Receptive Field)

#### Gradient Map (Shape: (2, 2))

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.0018 | 0.0018 | 
| [1] | 0.0018 | 0.0018 | 

### General Gradient Formula

$\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \frac{1}{k_h \cdot k_w} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}}$



## Layer: conv3 (Conv2d)

- **Input Shape**: torch.Size([1, 4, 2, 2])
- **Output Shape**: torch.Size([1, 2, 2, 2])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Computing output at position: (batch=0, out_channel=0, y=1, x=1)

#### Filter Weights (Shape: (4, 1, 1))

**Channel 0**

| Index | [0] | 
| --- | --- | 
| [0] | 0.1034 | 

**Channel 1**

| Index | [0] | 
| --- | --- | 
| [0] | 0.3807 | 

**Channel 2**

| Index | [0] | 
| --- | --- | 
| [0] | 0.1710 | 

**Channel 3**

| Index | [0] | 
| --- | --- | 
| [0] | 0.4088 | 

#### Input Receptive Field (Shape: (1, 1, 4))

**Channel 0**

| Index | [0] | [1] | [2] | [3] | 
| --- | --- | --- | --- | --- | 
| [0] | 1.4693 | 0.8798 | 0.0000 | 0.0000 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(1.4693 \times 0.1034) + (0.8798 \times 0.3807) + (0.0000 \times 0.1710) + (0.0000 \times 0.4088) + 0.2806$

### Computation Result

Calculated value: 0.767475

Actual output value: 0.767475

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

### Backward Computation for Conv2d at Position: (batch=0, out_channel=0, y=1, x=1)

#### Gradient Output Value

0.069219

#### Filter Weights

#### Filter Weights (Shape: (4, 1, 1))

**Channel 0**

| Index | [0] | 
| --- | --- | 
| [0] | 0.1034 | 

**Channel 1**

| Index | [0] | 
| --- | --- | 
| [0] | 0.3807 | 

**Channel 2**

| Index | [0] | 
| --- | --- | 
| [0] | 0.1710 | 

**Channel 3**

| Index | [0] | 
| --- | --- | 
| [0] | 0.4088 | 

#### Gradient Propagation to Input

| Position (y, x, c) | Formula | Gradient Value |
| ----------------- | ------- | -------------- |
| (1, 1, 0) | $(0.0692 \times 0.1034)$ | 0.007160 |
| (1, 1, 1) | $(0.0692 \times 0.3807)$ | 0.026349 |
| (1, 1, 2) | $(0.0692 \times 0.1710)$ | 0.011834 |
| (1, 1, 3) | $(0.0692 \times 0.4088)$ | 0.028297 |

#### Weight Gradient Computation

| Weight Position (out_c, in_c, ky, kx) | Formula | Gradient Value |
| ------------------------------------ | ------- | -------------- |
| (0, 0, 0, 0) | $(0.0692 \times 1.4693)$ | 0.101706 |
| (0, 1, 0, 0) | $(0.0692 \times 0.8798)$ | 0.060898 |
| (0, 2, 0, 0) | $(0.0692 \times 0.0000)$ | 0.000000 |
| (0, 3, 0, 0) | $(0.0692 \times 0.0000)$ | 0.000000 |

#### Bias Gradient Computation

Bias Gradient for output channel 0: 0.069219

### General Gradient Formulas

#### Input Gradient Formula

$\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

#### Weight Gradient Formula

$\frac{\partial L}{\partial w_{c_{out},c_{in},k_h,k_w}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot x_{n,c_{in},h_{in}+k_h,w_{in}+k_w}$

#### Bias Gradient Formula

$\frac{\partial L}{\partial b_{c_{out}}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}}$



## Layer: conv_out (Conv2d)

- **Input Shape**: torch.Size([1, 2, 2, 2])
- **Output Shape**: torch.Size([1, 2, 1, 1])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Computing output at position: (batch=0, out_channel=0, y=0, x=0)

#### Filter Weights (Shape: (2, 2, 2))

**Channel 0**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.3106 | -0.1264 | 
| [1] | -0.2203 | -0.0560 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.1043 | 0.2032 | 
| [1] | -0.0275 | 0.1508 | 

#### Input Receptive Field (Shape: (2, 2, 2))

**Channel 0**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.6659 | 0.0000 | 
| [1] | 0.5708 | 0.0000 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.7313 | 0.0000 | 
| [1] | 0.7675 | 0.0000 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(0.6659 \times 0.3106) + (0.5708 \times -0.1264) + (0.7313 \times -0.2203) + (0.7675 \times -0.0560) + (0.0000 \times 0.1043) + (0.0000 \times 0.2032) + (0.0000 \times -0.0275) + (0.0000 \times 0.1508) + -0.3441$

### Computation Result

Calculated value: -0.413463

Actual output value: -0.413463

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

### Backward Computation for Conv2d at Position: (batch=0, out_channel=0, y=0, x=0)

#### Gradient Output Value

-0.750153

#### Filter Weights

#### Filter Weights (Shape: (2, 2, 2))

**Channel 0**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.3106 | -0.1264 | 
| [1] | -0.2203 | -0.0560 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.1043 | 0.2032 | 
| [1] | -0.0275 | 0.1508 | 

#### Gradient Propagation to Input

| Position (y, x, c) | Formula | Gradient Value |
| ----------------- | ------- | -------------- |
| (0, 0, 0) | $(-0.7502 \times 0.3106)$ | -0.233020 |
| (0, 1, 0) | $(-0.7502 \times -0.1264)$ | 0.094843 |
| (1, 0, 0) | $(-0.7502 \times -0.2203)$ | 0.165264 |
| (1, 1, 0) | $(-0.7502 \times -0.0560)$ | 0.042000 |
| (0, 0, 1) | $(-0.7502 \times 0.1043)$ | -0.078245 |
| (0, 1, 1) | $(-0.7502 \times 0.2032)$ | -0.152427 |
| (1, 0, 1) | $(-0.7502 \times -0.0275)$ | 0.020658 |
| (1, 1, 1) | $(-0.7502 \times 0.1508)$ | -0.113148 |

#### Weight Gradient Computation

| Weight Position (out_c, in_c, ky, kx) | Formula | Gradient Value |
| ------------------------------------ | ------- | -------------- |
| (0, 0, 0, 0) | $(-0.7502 \times 0.6659)$ | -0.499529 |
| (0, 0, 0, 1) | $(-0.7502 \times 0.5708)$ | -0.428204 |
| (0, 0, 1, 0) | $(-0.7502 \times 0.7313)$ | -0.548584 |
| (0, 0, 1, 1) | $(-0.7502 \times 0.7675)$ | -0.575724 |
| (0, 1, 0, 0) | $(-0.7502 \times 0.0000)$ | -0.000000 |
| (0, 1, 0, 1) | $(-0.7502 \times 0.0000)$ | -0.000000 |
| (0, 1, 1, 0) | $(-0.7502 \times 0.0000)$ | -0.000000 |
| (0, 1, 1, 1) | $(-0.7502 \times 0.0000)$ | -0.000000 |

#### Bias Gradient Computation

Bias Gradient for output channel 0: -0.750153

### General Gradient Formulas

#### Input Gradient Formula

$\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

#### Weight Gradient Formula

$\frac{\partial L}{\partial w_{c_{out},c_{in},k_h,k_w}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot x_{n,c_{in},h_{in}+k_h,w_{in}+k_w}$

#### Bias Gradient Formula

$\frac{\partial L}{\partial b_{c_{out}}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}}$

