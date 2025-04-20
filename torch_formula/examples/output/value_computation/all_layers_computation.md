# Detailed Analysis of All Layers in TinyCNN Model

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
| [0] | -0.0564 | 0.2138 | -0.0643 | 
| [1] | -0.1281 | 0.0525 | 0.1127 | 
| [2] | 0.0742 | 0.2271 | -0.2033 | 

**Channel 1**

| Index | [0] | [1] | [2] | 
| --- | --- | --- | --- | 
| [0] | -0.1295 | -0.2122 | 0.0403 | 
| [1] | 0.0277 | 0.1636 | -0.0793 | 
| [2] | 0.1491 | 0.2250 | -0.1543 | 

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

$(0.5800 \times -0.0564) + (1.3000 \times 0.2138) + (1.3686 \times -0.0643) + (1.4657 \times -0.1281) + (1.8643 \times 0.0525) + (2.2229 \times 0.1127) + (1.2486 \times 0.0742) + (2.0186 \times 0.2271) + (1.9471 \times -0.2033) + (0.3629 \times -0.1295) + (0.9214 \times -0.2122) + (0.9700 \times 0.0403) + (0.9514 \times 0.0277) + (1.2243 \times 0.1636) + (1.3929 \times -0.0793) + (0.9400 \times 0.1491) + (1.4814 \times 0.2250) + (1.3814 \times -0.1543) + 0.1739$

### Computation Result

Calculated value: 0.820055

Actual output value: 0.820055

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$



## Layer: pool2 (AvgPool2d)

- **Input Shape**: torch.Size([1, 4, 4, 4])
- **Output Shape**: torch.Size([1, 4, 2, 2])

### Forward Pass

- **General Formula**: $y_{n,c,h_{out},w_{out}} = \frac{1}{k_h \cdot k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}$

### Computing output at position: (batch=0, channel=0, y=1, x=1)

#### Input Receptive Field (Shape: (2, 2))

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.7004 | 1.2385 | 
| [1] | 0.1651 | 0.0877 | 

Kernel size: (2, 2)

### General Formula

$y_{n,c,h_{out},w_{out}} = \frac{1}{k_h \cdot k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}$

### Value Substitution

$\frac1{2 \times 2} \times (0.7004 + 1.2385 + 0.1651 + 0.0877)$

### Computation Result

Calculated average value: 0.547938

Actual output value: 0.547938

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \frac{1}{k_h \cdot k_w} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}}$



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
| [0] | 0.3082 | 

**Channel 1**

| Index | [0] | 
| --- | --- | 
| [0] | -0.3845 | 

**Channel 2**

| Index | [0] | 
| --- | --- | 
| [0] | -0.1656 | 

**Channel 3**

| Index | [0] | 
| --- | --- | 
| [0] | -0.2766 | 

#### Input Receptive Field (Shape: (1, 1, 4))

**Channel 0**

| Index | [0] | [1] | [2] | [3] | 
| --- | --- | --- | --- | --- | 
| [0] | 0.5479 | 0.1062 | 0.4359 | 0.3585 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(0.5479 \times 0.3082) + (0.1062 \times -0.3845) + (0.4359 \times -0.1656) + (0.3585 \times -0.2766) + -0.4827$

### Computation Result

Calculated value: -0.526049

Actual output value: -0.526049

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$



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
| [0] | 0.2235 | -0.3527 | 
| [1] | 0.1200 | -0.2583 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.0408 | -0.2911 | 
| [1] | 0.3003 | -0.0986 | 

#### Input Receptive Field (Shape: (2, 2, 2))

**Channel 0**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.0000 | 0.4418 | 
| [1] | 0.0000 | 0.6091 | 

**Channel 1**

| Index | [0] | [1] | 
| --- | --- | --- | 
| [0] | 0.0000 | 0.6356 | 
| [1] | 0.0000 | 0.6736 | 

### General Formula

$y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Value Substitution

$(0.0000 \times 0.2235) + (0.0000 \times -0.3527) + (0.0000 \times 0.1200) + (0.0000 \times -0.2583) + (0.4418 \times 0.0408) + (0.6091 \times -0.2911) + (0.6356 \times 0.3003) + (0.6736 \times -0.0986) + -0.0094$

### Computation Result

Calculated value: -0.044280

Actual output value: -0.044280

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

