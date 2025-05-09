<p align="center">
<img style="width:300px;max-width:300px;" src="./logo.png"/>
</p>

`patches` is an experiment (first and foremost). 

Here's what the author is trying to understand:
* How does staged metaprogramming work in Mojo?
* Can staged metaprogramming operate on a kernel DSL (similar to `jax.pallas`, or Nvidia's `warp` library)?
* Can the kernel DSL faithfully reflect "enough" of the low-level details of e.g. layouts & tiling to be usefully used as a substrate for writing kernels, differentiating them, doing other spooky things with them, etc.

To answer this question, the author is roughly following this set of steps:
* Encode a "JAX-like" `Expr` representation _at comptime_ in Mojo. Write a tracing process that allows for construction of this representation out of `Mojo` `fn` types.
* Extend the abstract lattice that JAX presents (`dtype` and `shape`) with "richer" information (reflecting underlying tensor information like layouts).

Here's the first concrete goal the author has set for himself:
* Express matmul as a program in the staged DSL, and differentiate it (aim high!)