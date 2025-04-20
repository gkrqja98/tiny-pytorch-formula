# Detailed Computation for Conv2d Layer 'conv2' (Direct Calculation)

## Position: (0, 0, 2, 2)

### Computing output at position: (batch=0, out_channel=0, y=2, x=2)

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

$(1.8643 \times -0.0564) + (2.2229 \times 0.2138) + (1.9200 \times -0.0643) + (2.0186 \times -0.1281) + (1.9471 \times 0.0525) + (1.8729 \times 0.1127) + (1.7957 \times 0.0742) + (2.0714 \times 0.2271) + (1.8914 \times -0.2033) + (1.2243 \times -0.1295) + (1.3929 \times -0.2122) + (1.1829 \times 0.0403) + (1.4814 \times 0.0277) + (1.3814 \times 0.1636) + (1.3857 \times -0.0793) + (1.0443 \times 0.1491) + (1.3814 \times 0.2250) + (1.3686 \times -0.1543) + 0.1739$

### Computation Result

Calculated value: 0.700424

Actual output value: 0.700424

