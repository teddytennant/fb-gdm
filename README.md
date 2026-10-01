# fb_gdm

Variational conditional score for one reverse step of a DDPM, for a linear observation y = A x0 + noise. JAX.

## What is implemented

Corollary 1 (dense posterior mean and covariance, and the conditional score), Proposition 2 (conjugate CAVI), Corollary 2 (separable matvec updates with a 1D step that maximizes the free energy), and Algorithm 2 (one reverse step). The step size is a 1D Newton maximizer of F on (0, 1]; the closed form cited from Fraysse and Rodet is not in the paper. Gamma shapes are m/2 and n/2 because the prior offsets are zero. Rates are floored at 1e-12.

## What does not match the paper

No image experiments, no pretrained DDPM. K is a caller argument, the paper does not state an iteration count. Gamma shapes are m/2 and n/2 (prior offsets zero).

## Layout

`src/fb_gdm/` is the package. `tests/` holds the numerical checks. Install with the extra that pins pytest:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[tests]"
.venv/bin/python -m pytest -q
```
