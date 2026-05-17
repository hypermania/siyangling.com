---
title: 'How CUDA Graph is used in BlackholePerturbations'
date: 2026-04-29
permalink: /posts/2026/04/cuda-graph-blackholeperturbations/
tags:
  - CUDA
  - CUDA Graph
  - C++
  - numerical relativity
  - black holes
---

The CUDA Graph support in `hypermania/BlackholePerturbations` is centered on the class `CudaTeukolskyScalarPDE` in `src/teukolsky_cubic_cuda.cu` and `src/teukolsky_cubic_cuda.cuh`. It was introduced in commit `8a86f7c` ("Successfully tested CudaTeukolskyScalarPDE, implemented with CUDA graph"). The point of the implementation is not to change the Teukolsky evolution equations themselves, but to reduce the CPU-side overhead of repeatedly launching the same GPU work pattern during time integration.

The key observation is that the solver evaluates the same right-hand-side structure many times. The field values change from call to call, but the dependency graph among kernels does not. CUDA Graph therefore fits the workload well: the code builds the launch DAG once, instantiates it as an executable graph, and reuses it whenever Odeint asks for another RHS evaluation.

Physics setting
======

The CUDA Graph implementation discussed here supports the Teukolsky equation calculations in [arXiv:2603.20379](https://arxiv.org/abs/2603.20379) and the Schwarzschild calculations in [arXiv:2503.19967](https://arxiv.org/abs/2503.19967). The physical problem is the late-time nonlinear tail of black hole perturbation ringdown.

The Teukolsky equation governs field perturbations of spin weight $s$ around a Kerr black hole. After a change of variable $\tilde\psi \equiv (\Delta^s r) \psi$ and projection onto spin-weighted spherical harmonics ${}_s Y_{\ell m}$, the equation reduces to a system of coupled 1+1D PDEs for the mode amplitudes $\tilde\psi_{\ell m}(t, r)$:

$$
\partial_t^2 \tilde\psi_{\ell m} = -\sum_{\ell'm'} \left(
C_{\ell'm'\ell m} \, \tilde\psi_{\ell'm'}
+ C^{(t)}_{\ell'm'\ell m} \, \partial_t \tilde\psi_{\ell'm'}
+ C^{(r)}_{\ell'm'\ell m} \, \partial_r \tilde\psi_{\ell'm'}
+ C^{(rr)}_{\ell'm'\ell m} \, \partial_r^2 \tilde\psi_{\ell'm'}
\right)
$$

The coefficients $C^{(\cdot)}_{\ell'm'\ell m}(r)$ are fixed by the Kerr parameters $M$ and $a$. Axisymmetry requires $C^{(\cdot)}_{\ell'm'\ell m} = 0$ for $m \neq m'$; for $a=0$, spherical symmetry further decouples all modes. The system is first-order reduced by promoting $\partial_t \tilde\psi_{\ell m}$ to a dynamical variable, forming the state vector $x = [\psi_{\ell m}, \partial_t \psi_{\ell m}]$.

This structure fixes the shape of the GPU workload. Radial derivatives are computed per mode, then each output harmonic assembles its second time derivative from a sum of coefficient-times-field terms drawn from four coupling maps (for $\psi$, $\partial_t\psi$, $\partial_r\psi$, $\partial_r^2\psi$). The number of contributing terms varies by harmonic mode. CUDA Graph was introduced because this coupled operator is structurally fixed but launched many times during the ODE evolution.

What the graph is accelerating
======

The state is stored as a thrust device vector of complex values,

`x = [psi_{lm}, dt psi_{lm}]`,

with one radial grid segment for each spherical-harmonic mode. For every call to `operator()(x, dxdt, t)`, the code must:

1. compute `dr psi_{lm}`,
2. compute `drdr psi_{lm}`,
3. assemble `d^2 psi / dt^2` for each output mode using precomputed coupling maps,
4. copy the first-order time-derivative block from `x` into the first half of `dxdt`,
5. optionally add the cubic self-interaction contribution,
6. optionally add an external source term.

This is naturally expressed as many small kernel launches rather than one giant kernel. For the common `l_max = 5` configuration used in `main.cpp`, there are `(l_max + 1)^2 = 36` harmonic modes. The linear graph therefore contains:

- 36 kernels for `drdr`,
- 36 kernels for `dr`,
- 1 barrier node,
- 36 kernels for the second-time-derivative assembly,
- 1 device-to-device memcpy node for copying `dt psi`.

That is 110 nodes before the nonlinear term is even included. When `lambda != 0`, the code launches a second graph with another 73 nodes to compute the cubic term. Since `boost::numeric::odeint::integrate_adaptive` invokes the RHS many times, launch overhead becomes worth optimizing.

<figure>
  <img src="/images/cuda-graph-blackholeperturbations-compute-graph.svg" alt="Schematic CUDA graph for the Teukolsky right-hand-side calculation">
  <figcaption>Figure 1. Schematic compute graph for one right-hand-side evaluation. The implementation creates one derivative and assembly node per harmonic mode, inserts graph barriers where scratch buffers must be complete, caches executable graphs by device pointer pair, and keeps the explicitly time-dependent source outside the graph.</figcaption>
</figure>

Why CUDA Graph was a good fit
======

CUDA Graph fits this solver because the launch topology is stable.

The coupling maps are fixed once `Param` is chosen. The number of modes is fixed. The scratch buffers are persistent members of `CudaTeukolskyScalarPDE`. From one RHS evaluation to the next, only the data stored in those buffers changes.

That makes the workload very different from an irregular GPU pipeline. Here the expensive part on the host is repeatedly submitting the same dependency structure. CUDA Graph removes most of that repetition without requiring a large refactor of the numerical code.

There is also a software-engineering reason this was attractive. The code already had a clean decomposition into physically meaningful kernels: derivatives, assembly of the linear operator, and nonlinear mode coupling. CUDA Graph lets the implementation keep that structure and optimize scheduling around it, instead of forcing everything into a monolithic fused kernel.

How the main graph is built
======

The main graph is constructed in `CudaTeukolskyScalarPDE::prepare_cuda_graph(const State &x, State &dxdt)`.

The constructor first precomputes the coupling coefficients on the host, copies them into device vectors, and allocates persistent device scratch space such as `dr_psi_lm` and `drdr_psi_lm`. This matters because graph nodes store raw pointers into those buffers.

The graph construction then proceeds in four stages.

First, for each harmonic `lm`, the code adds one node for `CUDAKernel::drdr_complex_double_kernel` and one node for `CUDAKernel::dr_complex_double_kernel`. These kernels write the second and first radial derivatives of the corresponding mode.

Second, it inserts an empty barrier node with `cudaGraphAddEmptyNode`. All derivative nodes feed into this barrier, and all later assembly nodes depend on it. This is the graph-level representation of the data dependency that every `d^2 psi / dt^2` evaluation needs the derivative buffers to be complete first.

Third, it loops over each output harmonic and assembles the right-hand side of the Teukolsky equation from four coupling maps:

- `psi_lm_map`,
- `dt_psi_lm_map`,
- `dr_psi_lm_map`,
- `drdr_psi_lm_map`.

For the selected harmonic, the code flattens coefficient pointers and field pointers into an argument array and dispatches a specialized kernel from

`CUDAKernel::assign_lhs_2terms_complex_double_kernels[num_terms]`.

The kernel family `assign_lhs_2terms_complex_double_kernels` is a vector of 21 specialized CUDA kernels, indexed 0 through 20. Each kernel has a fixed number of `(coefficient, field)` pointer pairs and computes, on a per-grid-point basis:

```cuda
// Generic form of the kernel family (k = 0, ..., 20)
__global__
void assign_lhs_2terms_complex_double_kernel_k(
    thrust::complex<double> * __restrict__ lhs_ptr,
    const thrust::complex<double> * __restrict__ term1_ptr_0,
    const thrust::complex<double> * __restrict__ term2_ptr_0,
    // ...  (2k - 2) additional term1/term2 pointer pairs ...
    const thrust::complex<double> * __restrict__ term1_ptr_{k-1},
    const thrust::complex<double> * __restrict__ term2_ptr_{k-1},
    const int grid_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < grid_size){
    lhs_ptr[i] = - term1_ptr_0[i] * term2_ptr_0[i]
                 - term1_ptr_1[i] * term2_ptr_1[i]
                 - ...
                 - term1_ptr_{k-1}[i] * term2_ptr_{k-1}[i];
  }
}
```

Here `term1` pointers point to radial coupling coefficients $C^{(\cdot)}(r)$ and `term2` pointers point to the corresponding field slices ($\tilde\psi$, $\partial_t\tilde\psi$, $\partial_r\tilde\psi$, $\partial_r^2\tilde\psi$). The product at each grid point computes one term in the sum on the right-hand side of the Teukolsky equation above.

Why 21 fixed-arity kernels rather than a single kernel with a loop? The per-mode Teukolsky equation has a variable number of coupling terms: different $(\ell,m)$ modes draw from coupling maps of different sizes. The number of terms `num_terms` is known at graph-construction time but varies across modes:

```cpp
const size_t num_terms = args.size() / 2;
node_params.func = (void *)CUDAKernel::assign_lhs_2terms_complex_double_kernels[num_terms];
```

Since CUDA Graph kernel nodes must have their function pointer fixed at graph-instantiation time, a runtime loop over terms cannot be captured in the graph. The specialized-arity kernel family solves this: each variant is a plain kernel with a compile-time-fixed argument list, suitable for embedding in a `cudaKernelNodeParams`. The dispatch is dynamic at graph-construction time but static at graph-launch time.

Fourth, the first half of `dxdt` is filled from the second half of `x`. When `ko_epsilon == 0`, this is a `cudaGraphAddMemcpyNode1D` device-to-device copy. When `ko_epsilon != 0`, the code instead launches `CUDAKernel::ko_and_copy_complex_double_kernel` once per mode so that copying and Kreiss-Oliger dissipation happen together.

How the nonlinear graph is built
======

If `param.lambda != 0`, `operator()` launches a second graph prepared by `prepare_lambda_cuda_graph`.

This graph computes the cubic term in two stages.

In the first stage, it forms each harmonic of `phi^2` into the scratch buffer `psi_sqr_lm`. The required mode-coupling data comes from `SPH::make_coupling_info_map(l_max)`. As in the linear graph, the code uses a family of specialized kernels indexed by the number of coupling terms.

In the second stage, after another barrier node, the graph multiplies `psi_sqr_lm` by the precomputed nonlinear coefficient tables `lambda_coeffs` and accumulates the result into the second-half block of `dxdt`.

Keeping the nonlinear term in a separate graph is a sensible design choice. It keeps the linear graph simple, and it avoids paying nonlinear graph setup or launch cost when `lambda = 0`.

Separating the nonlinear computation into two stages (product `phi^2` then accumulation with `lambda_coeffs`) is also algorithmically more efficient than summing all terms in a single stage. If the nonlinear terms were merged directly into the per-mode assembly, each output harmonic `(l,m)` would need terms for every pair of input modes `(l_1,m_1)` and `(l_2,m_2)` that couple to it—leading to a kernel with $O(N^3)$ terms, where $N = (l_\text{max}+1)^2$ is the number of harmonic modes. With the two-stage approach, the first stage computes each harmonic of `phi^2` with $O(N^2)$ terms (one per input pair), and the second stage accumulates the result with $O(N^2)$ terms. For $l_\text{max} = 5$, $N = 36$, and $N^3 / N^2 = 36$, so the two-stage approach eliminates a factor-of-36 blowup in the number of terms per kernel.

How graph reuse works
======

The runtime entry point is `operator()(const State &x, State &dxdt, const Scalar t)`. This functor is called by `boost::numeric::odeint::integrate_adaptive` with a `runge_kutta_dopri5` stepper. The stepper calls `operator()` at each Runge-Kutta stage within a timestep, and the stepper manages its own stage-wise GPU memory buffers for intermediate results.

The graph topology depends on where `x` and `dxdt` live in device memory, since the kernel-node parameters contain raw pointers into those buffers. But the stepper's internal buffer management means the caller does not know in advance what memory addresses `x` and `dxdt` will occupy at each stage—and modifying the stepper code to expose those addresses was not the chosen approach.

The solution is a pointer-keyed cache. The code extracts raw device pointers from `x` and `dxdt` and uses the pair `(x_ptr, dxdt_ptr)` as a key in `graph_exec_mapping` and `lambda_graph_exec_mapping`. If the key is new, the code builds a fresh `cudaGraph_t`, instantiates it with `cudaGraphInstantiate`, stores the resulting `cudaGraphExec_t`, and destroys the temporary graph object. If the key has been seen before, it skips reconstruction and directly calls `cudaGraphLaunch`.

The consequence of this design is that new graphs are instantiated whenever a new buffer pair appears. With a 4-stage Runge-Kutta scheme such as `dopri5`, the stepper uses distinct stage buffers, so up to 4 compute graphs are instantiated on the first timestep—one for each `(x, dxdt)` pair used by the stepper. On subsequent timesteps, the stepper recycles those same buffers, and all graph launches hit the cache without further instantiation.

What remains outside the graph
======

The external source term is still applied outside CUDA Graph:

```cpp
if(add_source){
  add_source(x, dxdt, t);
}
```

This is a good boundary. The source depends explicitly on the current time `t`, while the rest of the operator is mostly a fixed pointer-and-dependency structure. The codebase also contains commented-out experiments with `cudaGraphExecUpdate`, which suggests that more dynamic graph updating was considered but not adopted in the current implementation.

Strengths and limitations
======

The implementation has several strengths.

- It preserves the usual Odeint functor interface.
- It keeps the decomposition into physically meaningful kernels.
- It makes dependencies explicit through graph barrier nodes.
- It reuses persistent device buffers and precomputed coupling metadata.
- It separates the nonlinear graph from the linear graph cleanly.

The main limitations are also clear.

- The graph cache is keyed by raw pointers and is not explicitly destroyed in the class, so many transient buffers would create a resource leak.
- The design assumes pointer-stable storage, which is true for the intended solver path but not universally guaranteed in more dynamic workflows.
- CUDA Graph reduces launch overhead, but it does not change the fact that the operator still has many mode-level nodes.
- The source term remains outside graph capture, so the full RHS is not one single graph launch.

Bottom line
======

CUDA Graph is used in `BlackholePerturbations` as a scheduling optimization for a fixed-structure GPU PDE operator. The numerical method is still the same Teukolsky evolution scheme; what changes is how the large collection of repeated GPU launches is submitted.

That is why CUDA Graph makes sense here. The solver has a stable DAG, persistent buffers, repeated RHS evaluations, and many small kernels. Instead of redesigning the physics code around one fused kernel, the implementation captures the existing computation graph and replays it efficiently.
