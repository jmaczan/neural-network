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

now another derivative of $S_i$ w.r.t $z_k$ when $k \ne i$:

$$
\frac{\partial{S_i}}{\partial{z_k}}=\frac{\partial}{\partial{z_k}}(\frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}})
$$

once again let's apply the quotient rule of differentiation

$$
\frac{\partial}{\partial{z_k}}(\frac{e^{z_i}}{\sum_{j=1}^K{e^{z_j}}})=\frac{\frac{\partial{e^{z_i}}}{\partial{z_k}}\sum_{j=1}^K{e^{z_j}}-e^{z_i}\frac{\partial \sum_{j=1}^K{e^{z_j}}}{\partial{z_k}}}{(\sum_{j=1}^K{e^{z_j}})^2}
$$

first term in numerator is $0$, because from the basic principles of partial differentiation, where the derivative of a function with respect to a variable that does not appear in the function is $0$

then there's a sum, from which all terms are $0$ except $e^{z_k}$. so then:

$$
\frac{-e^{z_i}e^{z_k}}{(\sum_{j=1}^K{e^{z_j}})^2}=S_i\frac{-e^{z_k}}{\sum_{j=1}^K{e^{z_j}}}
$$

So $\frac{-e^{z_k}}{\sum_{j=1}^K{e^{z_j}}}$ is softmax but with ${z_k}$ as a parameter,so then:

$$
\frac{\partial{S_i}}{\partial{z_k}}=-S_i \cdot S_k
$$
