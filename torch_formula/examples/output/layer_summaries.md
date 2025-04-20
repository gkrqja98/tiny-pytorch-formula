# TinyCNN Layer Summaries

## Layer: conv1 (Conv2d)

- **Input Shape**: torch.Size([1, 1, 8, 8])
- **Output Shape**: torch.Size([1, 2, 8, 8])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$



## Layer: pool1 (MaxPool2d)

- **Input Shape**: torch.Size([1, 2, 8, 8])
- **Output Shape**: torch.Size([1, 2, 4, 4])

### Forward Pass

- **General Formula**: $y_{n,c,h_{out},w_{out}} = \max_{k_h,k_w} x_{n,c,h_{in}+k_h,w_{in}+k_w}$

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \begin{cases} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}} & \text{if } x_{n,c,h_{in},w_{in}} \text{ is max in pool} \\ 0 & \text{otherwise} \end{cases}$



## Layer: conv2 (Conv2d)

- **Input Shape**: torch.Size([1, 2, 4, 4])
- **Output Shape**: torch.Size([1, 4, 4, 4])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$



## Layer: pool2 (AvgPool2d)

- **Input Shape**: torch.Size([1, 4, 4, 4])
- **Output Shape**: torch.Size([1, 4, 2, 2])

### Forward Pass

- **General Formula**: $y_{n,c,h_{out},w_{out}} = \frac{1}{k_h \cdot k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}$

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c,h_{in},w_{in}}} = \frac{1}{k_h \cdot k_w} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c,h_{out},w_{out}}}$



## Layer: conv3 (Conv2d)

- **Input Shape**: torch.Size([1, 4, 2, 2])
- **Output Shape**: torch.Size([1, 2, 2, 2])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$



## Layer: conv_out (Conv2d)

- **Input Shape**: torch.Size([1, 2, 2, 2])
- **Output Shape**: torch.Size([1, 2, 1, 1])

### Forward Pass

- **General Formula**: $y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}$

### Backward Pass

- **General Formula**: $\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out},k_h,k_w} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$

