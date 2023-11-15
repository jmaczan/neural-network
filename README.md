# neural-network

🐱 Neural Network from a very scratch - backpropagation, gradient descent, activation functions

## prerequisities

install anaconda, so you have `conda` available in shell

## development

```
# To activate this environment, use
#
#     $ conda activate nn
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

export env settings to .yml:

```
conda env export --from-history > environment.yml
```

## notes

softmax derivative:

$$
softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}}
$$

we need two derivatives of softmax - with respect to $z_i$ and with respect to $z_k$ when $k \ne i$

let's say $softmax(z_i) = S_i$. derivative of $S_i$ w.r.t $z_i$:

$$
\frac{\partial{S_i}}{\partial{z_i}}=\frac{\partial}{\partial{z_i}}(\frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}})
$$

apply the quotient rule of differentiation

$$
\frac{\partial}{\partial {x}}(\frac{f}{g})=\frac{f'g - fg'}{g^2}
$$

so then

$$
\frac{\partial}{\partial{z_i}}(\frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}})=\frac{e^{z_i}\sum_{j=1}^K{e^{z_j}} - e^{z_i}\sum_{j=1}^K{e^{z_j}}'}{(\sum_{j=1}^K{e^{z_j}})^2}
$$

derivative of sum ${\sum_{j=1}^K{e^{z_j}}}$ is ${e^{z_i}}$, because all other derivatives of $e^(z_k)$ w.r.t $e^{z_i}$ are $0$

$$
\frac{e^{z_i}\sum_{j=1}^K{e^{z_j}} - e^{z_i}e^{z_i}}{(\sum_{j=1}^K{e^{z_j}})^2} = \frac{e^{z_i}(\sum_{j=1}^K{e^{z_j}} - e^{z_i})}{(\sum_{j=1}^K{e^{z_j}})^2}=\frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}}\frac{(\sum_{j=1}^K{e^{z_j}})-e^{z_i}}{\sum_{j=1}^K{e^{z_j}}}=S_i(1-S_i)
$$
