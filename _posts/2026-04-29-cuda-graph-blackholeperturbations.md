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

The part of the code discussed here was written for the Kerr black hole calculations in [arXiv:2603.20379](https://arxiv.org/abs/2603.20379), "Nonlinear tails in the Kerr black hole ringdown". The broader `BlackholePerturbations` repository also contains the Schwarzschild calculations used in [arXiv:2503.19967](https://arxiv.org/abs/2503.19967), "Dynamical nonlinear tails in Schwarzschild black hole ringdown".

The physical question is the late-time behavior of black hole perturbations. After a compact disturbance, the waveform first contains the familiar quasinormal ringing of the black hole. At sufficiently late times, however, the signal is controlled by power-law tails. Linear perturbation theory already predicts such tails, but nonlinearities can source additional tails with different amplitudes and decay laws. These nonlinear tails are interesting because they probe mode coupling in the strong-field problem and may become the dominant part of the late waveform.

For Kerr, the natural equation for this calculation is the Teukolsky equation. The code evolves spin-weighted fields decomposed into spherical harmonic modes up to a chosen `l_max`, with mode-mode coupling coefficients computed from the background Kerr parameters. In the sourced calculations, the source falls as `r^{-beta}` in the far field. In the scalar self-interaction runs, the nonlinear dynamics are represented by an effective `lambda psi^2` source. The paper derives and numerically checks the far-field nonlinear-tail power law `t^{-ell-beta-s}` for spin weight `s`, harmonic mode `ell m`, and source falloff `beta`, and it also studies the dynamical formation of nonlinear tails for a massless scalar.

This physics explains the shape of the GPU workload. Each right-hand-side evaluation is not just a one-dimensional stencil. It is a coupled harmonic calculation: radial derivatives are computed mode by mode, then each output harmonic is assembled from field, time-derivative, first-radial-derivative, and second-radial-derivative contributions across the coupling maps. The optional scalar nonlinearity adds another all-mode product and accumulation stage. CUDA Graph was introduced because this coupled operator is structurally fixed but launched many times during the ODE evolution.

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

The number of terms varies from mode to mode, so the kernel family is indexed dynamically.

Fourth, the first half of `dxdt` is filled from the second half of `x`. When `ko_epsilon == 0`, this is a `cudaGraphAddMemcpyNode1D` device-to-device copy. When `ko_epsilon != 0`, the code instead launches `CUDAKernel::ko_and_copy_complex_double_kernel` once per mode so that copying and Kreiss-Oliger dissipation happen together.

How the nonlinear graph is built
======

If `param.lambda != 0`, `operator()` launches a second graph prepared by `prepare_lambda_cuda_graph`.

This graph computes the cubic term in two stages.

In the first stage, it forms each harmonic of `phi^2` into the scratch buffer `psi_sqr_lm`. The required mode-coupling data comes from `SPH::make_coupling_info_map(l_max)`. As in the linear graph, the code uses a family of specialized kernels indexed by the number of coupling terms.

In the second stage, after another barrier node, the graph multiplies `psi_sqr_lm` by the precomputed nonlinear coefficient tables `lambda_coeffs` and accumulates the result into the second-half block of `dxdt`.

Keeping the nonlinear term in a separate graph is a sensible design choice. It keeps the linear graph simple, and it avoids paying nonlinear graph setup or launch cost when `lambda = 0`.

How graph reuse works
======

The runtime graph cache lives in `operator()(const State &x, State &dxdt, const Scalar t)`.

The code extracts raw device pointers from `x` and `dxdt` and uses the pair

`(x_ptr, dxdt_ptr)`

as the key for:

- `graph_exec_mapping`,
- `lambda_graph_exec_mapping`.

If a key has not been seen before, the code creates a fresh `cudaGraph_t`, instantiates it with `cudaGraphInstantiate`, stores the resulting `cudaGraphExec_t`, and destroys the temporary graph object. If the key is already present, it skips reconstruction and directly calls `cudaGraphLaunch`.

This means the executable graph is effectively specialized not just to the problem parameters, but also to the actual memory addresses of the state and derivative vectors. That is a pragmatic way to avoid patching node parameters after instantiation.

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
