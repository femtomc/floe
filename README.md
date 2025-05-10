<p align="center">
<img style="width:300px;max-width:300px;" src="./logo.png"/>
</p>

`patches` is an experiment (first and foremost) - don't use it!

## Goals

Here's what the author is trying to understand:
* How might staged metaprogramming work in Mojo via `alias` and comptime?
* Can staged metaprogramming operate on a kernel DSL (similar to `jax.pallas`, or Nvidia's `warp` library)?
* Can the staging language faithfully reflect _enough_ of the low-level details of e.g. layouts & tiling to be usefully used as a substrate for writing kernels, differentiating them, doing other spooky things with them, etc.

To answer this question, the author is roughly following this set of steps:
* Encode a "JAX-like" `Expr` representation _at comptime_ in Mojo. Write a tracing process that allows for construction of this representation out of `Mojo` `fn` types.
* Extend the abstract lattice that JAX presents (`dtype` and `shape`) with "richer" information (reflecting underlying tensor information like layouts).

Here's the MVP (aim high!) concrete goal the author has set for himself:
* Express matmul as a program in the staged DSL, and differentiate it.

## Project log

### 05/09/2025 (Compile time staging in Mojo)

At compile time, able to do the following:

```mojo
fn f(x: Tensor) -> Tensor:
    return x + x + x * x

alias expr = stage1[f](
    tensor[
        DType.float32,
        Layout.col_major(3, 4),
    ]()
)
print(expr.value())
```

which prints the simple first-order DSL program:
```
{ lambda %0:f32[((3, 4):(1, 3))] .
  %1:f32[((3, 4):(1, 3))] = add %0 %0
  %2:f32[((3, 4):(1, 3))] = mul %0 %0
  %3:f32[((3, 4):(1, 3))] = add %1 %2
  return %3 }
```

Note that this program is in a functional style (closely following JAX). Kernel DSLs do not look like this (they support mutable writes via "reference" or thread primitives). The author is thinking about this next.

Lots of things wrong here (writing something for the first time!), but a good start.

### 05/10/2025 (Meditations on kernel programming)

Meditated deeper on the difference between JAX's purely functional array programming model, and kernel DSLs. I'm hung up on the difference in UI between these two models. My understanding _today_ is that kernel DSLs often support _mutation_ idioms, and provide primitives and types designed to allow reading and writing from locations in memory, along with ways to customize _where values live in memory_, which is immensely important to maximize performance on parallel accelerators.

#### Mojo's support for GPU kernels 

As of _today_, I think of Mojo's GPU programming as exposing a sort of full-featured kernel programming language. Within kernel functions, users have access to thread and memory constructs (`gpu.id.block_idx`, `gpu.id.thread_idx`). Having written a function, a user may `DeviceContext.enqueue_function` it - which compiles the function and readies it to run on a given device. 

Enqueueing it requires specifying things like grids of thread blocks. When the function is run, the blocks are mapped to streaming multiprocessors (SM) (on the GPU device) for execution. When an SM is assigned a thread block, it breaks the block into _warps_ which are groups of 32 or 64 threads (depending on GPU). Warps seem like the fundamental unit of SIMT execution.

The last paragraph is generic across several kernel programming models, including Mojo and `pallas`. The details here are aligned with execution on target hardware, and presumably exposing these details allow users to take control of things which affect performance of their kernels.

#### Control over memory 

Perhaps the central design issue here is control over memory, and how threads read and write over memory. JAX fully hides this part of GPU programming from the user - an immensely ergonomic design choice from the perspective of someone who just wants to get a computation running on the GPU, and worry about memory or layout optimization later.

When users start to reach for low-level control, mutability re-enters the picture. Take `jax.pallas` for instance, which introduces a `Ref` type (representing a mutable buffer at a location somewhere in GPU memory). The goal of `jax.pallas` is to _restore_ access to the things which JAX has abstracted away, so that users may use this access to improve the performance of their programs.

Interestingly, `jax.pallas` integrates with JAX. How exactly does this work? From the `jax.pallas` documentation, I believe that JAX allocates memory for its arrays into high-bandwidth memory (HBM) - and then moves values from HBM to SRAM during kernel computations. This is immensely costly!

The main goal of _kernel fusion optimization_ (which is possibly the most important optimization that a compiler like XLA performs) is to reduce the back-and-forth data movement from HBM and SRAM by taking multiple kernels, and fusing them into a single kernel, meaning that multiple memory movement steps are eliminated.

The following picture from the `jax.pallas` documentation illustrates the memory hierarchy of a GPU:

<p align="center">
<img style="width:600px;max-width:600px;" src="./gpu_mem.svg">
</p>

When a `jax.pallas` function is invoked inside of a JAX computation, an explicit memory movement from GMEM to SMEM (SRAM?) is requested. From the documentation:

> [(`jax.pallas`) Hello world in Pallas](https://docs.jax.dev/en/latest/pallas/quickstart.html#hello-world-in-pallas)
>
> On GPU, `x_ref` corresponds to a value in high-bandwidth memory (HBM) and when we do `x_ref[...]` we are copying the value from HBM into static RAM (SRAM) (this is a costly operation generally speaking!). We then use GPU vector compute to execute the addition, then copy the resulting value in SRAM back to HBM.

What about intermediates which are materialized as part of the computation of a kernel?

> [(`jax.pallas`) Memory spaces](https://docs.jax.dev/en/latest/pallas/gpu/reference.html#memory-spaces)
>
> Finally, the lowest level memory space is the register memory. This is where every single value (i.e. JAX array) in a Pallas kernel will be located. If the compiler runs out of registers to store those arrays, it will insert spills, meaning that it will periodically store and reload values to memory. Those spills often introduce other significant performance degradations and so we recommend avoiding them. The warning messages about spills can be clearly seen in the ptxas messages during kernel compilation. To make them visible, run with MOSAIC_GPU_DUMP_PTXAS=1 in your environment.

Which seems to answer this questions: intermediates are materialized into register memory. If intermediates spill over, movement to other parts of the memory hierarchy are introduced, which are also costly.

#### JAX/XLA & MAX

Reflecting on the above: systems like JAX/XLA and MAX remove control from the user, to give them a convenient GPU programming model which doesn't require that they think carefully about memory movement or layout, etc.

These programming models allow users to write programs that represent a sequence of kernel computations, then it's up to the compiler to allocate and orchestrate memory. In the worst case, you could indeed write a compiler that just executed the sequence of kernels one after the other (moving results to and fro from HBM to SRAM). But the data movement is immensely costly - so all compilers (XLA & MAX) perform kernel fusion to attempt to reduce data movements.

#### Reflections on functional, effectful computation

After some reading (Pallas documentation, Warp, Dex, [Comparison of Futhark to Dex](https://futhark-lang.org/blog/2020-12-28-futhark-and-dex.html))

It occurs to me (naive as I am), that most of these languages circle the drain around _algebraic effects_ as a way to pair a purely functional model with a mutation UI.