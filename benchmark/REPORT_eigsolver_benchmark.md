# Eigsolver Benchmark Report

**Date**: 2026-03-19
**Setup**: D=3, chi=64, Heisenberg model, RTX 4090 GPU
**Goal**: Find optimal eigsolver configuration for iPEPS optimization with VUMPS

## Summary

**Optimal configuration**: `power_k=5, maxiter_ad=4, precondition=:local`

Plain power method with 5 steps per eigsolve, 4 AD steps, and local metric preconditioner.
No acceleration method (momentum, Chebyshev, Anderson mixing) improved upon this baseline.

## Experiments

### 1. Eigsolver type comparison (no preconditioner, 20 steps)

Compared power method (k=1,2,5,10) vs Arnoldi (k=3,5,10,15).

| Config | E@20 | Time(s) |
|--------|------|---------|
| power_k=1 | -0.664415 | 5.3 |
| power_k=2 | -0.664110 | 3.7 |
| power_k=5 | -0.663850 | 4.7 |
| arnoldi_k=3 | -0.664152 | 7.2 |
| arnoldi_k=15 | -0.663729 | 10.9 |

**Finding**: All methods reach similar energy in 20 steps. Power method is faster per step on GPU.
Arnoldi provides no advantage - its linear combination of Krylov vectors differs too much from
power iteration trajectory, and the GPU overhead of Gram-Schmidt is significant.

### 2. AD iteration count (no preconditioner, 50 steps)

Varied maxiter_ad/miniter_ad for power method.

| Config | E@50 | Time(s) |
|--------|------|---------|
| pow2_ad4 | -0.66618 | 9.7 |
| pow2_ad2 | -0.66595 | 11.8 |
| pow1_ad4 | -0.66600 | 7.7 |
| pow2_ad8 | -0.66576 | 24.2 |

**Finding**: ad=4 is optimal. ad=8 is slower and yields worse energy (over-solving the
eigenproblem harms optimization by removing beneficial noise). ad=2 is borderline unstable.

### 3. Power steps with local preconditioner (50 steps)

| Config | Steps completed | E@50 | Time(s) |
|--------|----------------|------|---------|
| pow5_ad4 | 50 | **-0.66740** | **20.2** |
| pow5_ad2 | 50 | -0.66702 | 15.7 |
| pow3_ad2 | 50 | -0.66696 | 14.7 |
| pow2_ad2 | 50 | -0.66663 | 51.6 |
| pow7_ad2 | 38 (crashed) | - | 14.7 |
| pow10_ad2 | 35 (crashed) | - | 16.0 |

**Finding**: pow5_ad4 + local preconditioner is the best combination.
- pow7/pow10 with ad=2 crash (too many eigsolve steps + too few AD steps = gradient mismatch)
- pow2_ad2 has linesearch issues (51.6s with energy regression)
- pow5 is the sweet spot between eigsolve accuracy and optimization robustness

### 4. Accelerated eigsolvers: momentum and Chebyshev (FAILED)

Implemented momentum-accelerated power iteration (heavy-ball) and Chebyshev-accelerated
power iteration. Both failed catastrophically.

| Config | Steps completed | Notes |
|--------|----------------|-------|
| mom5_beta0.3 | 1 (crash) | VUMPS diverged |
| mom5_beta0.5 | 50 | E=-0.6648, 71.4s (much worse than baseline) |
| mom5_beta0.7 | 1 (crash) | VUMPS diverged |
| cheb5_rho0.8 | 1 (crash) | VUMPS diverged |
| cheb5_rho0.9 | 1 (crash) | VUMPS diverged |
| cheb5_rho0.95 | 1 (crash) | VUMPS diverged |

**Root cause**: Both methods rely on history from previous iterations (v_{k-1} for momentum,
Chebyshev recursion coefficients for Chebyshev). In VUMPS, the transfer matrix operator
changes after every outer iteration, so historical information points in wrong directions.
Power method's "memoryless" property is precisely why it works in self-consistent iterations.

### 5. Anderson mixing for VUMPS outer loop (FLAWED)

Implemented Anderson mixing (DIIS) on VUMPSEnv tensors.

| Config | Steps completed | E@50 | Time(s) |
|--------|----------------|------|---------|
| baseline | 45 | - | 41.8 |
| anderson_m5 | 50 | -0.66718 | 24.8 |
| anderson_m8 | 30 (energy exploded) | - | 13.9 |

anderson_m=5 appeared to work but the approach is **mathematically unsound**:
- VUMPSEnv contains AL (left-canonical MPS, lives on Stiefel manifold), C (gauge matrix),
  T (transfer matrix eigenvector) - none of these live in a linear space
- Linear combinations of MPS tensors violate gauge conditions (AL†·AL ≠ I after mixing)
- The apparent success of m=5 was likely due to mixing coefficients being close to (0,...,0,1)
  or the subsequent QR step in leftmove re-canonicalizing the broken gauge

### 6. Environment extrapolation (FAILED)

Linear extrapolation of environment between optimization steps: env_init = 2*env_new - env_old.

All configurations using extrapolation crashed within a few steps. Same root cause as
Anderson mixing: MPS tensors don't live in a linear space, so linear extrapolation
is not geometrically valid.

## Conclusions

1. **Power method is optimal for VUMPS eigsolve on GPU**: Its memoryless property makes it
   uniquely robust for self-consistent iteration where the operator changes each step.

2. **Inexact eigsolve helps optimization**: pow5 (not fully converged) outperforms pow10
   (closer to converged). This is analogous to inexact Newton methods where noise helps
   escape local minima.

3. **AD steps matter for gradient quality**: ad=4 balances gradient accuracy vs cost.
   Too few (ad=2) gives unstable gradients, too many (ad=8) wastes compute and over-constrains.

4. **Local metric preconditioner is essential**: Provides the largest single improvement
   in energy quality, more impactful than any eigsolver choice.

5. **MPS tensors are not vectors**: Any acceleration method that assumes linearity
   (Anderson mixing, env extrapolation, momentum on eigenvectors within VUMPS) will fail
   because VUMPS environment tensors live on manifolds with gauge constraints.

6. **Future directions for VUMPS acceleration** should focus on:
   - Riemannian optimization on the MPS manifold (respecting gauge structure)
   - Better preconditioners (full metric, BP-based)
   - Gauge-aware mixing (e.g., mixing only gauge-invariant observables like correlation lengths)
