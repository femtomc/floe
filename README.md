<p align="center">
<img style="width:300px;max-width:300px;" src="./logo.png"/>
</p>

`patches` is an experiment (first and foremost) -- don't use it!

## Goals

Here's what the author is trying to understand:
* How might staged metaprogramming work in Mojo via `alias` and comptime?
* Can staged metaprogramming operate on a kernel DSL (similar to `jax.pallas`, or Nvidia's `warp` library)?
* Can the staging language faithfully reflect _enough_ of the low-level details of e.g. layouts & tiling to be usefully used as a substrate for writing kernels, differentiating them, doing other spooky things with them, etc.

To answer this question, the author is roughly following this set of steps:
* Encode a "JAX-like" `Expr` representation _at comptime_ in Mojo. Write a tracing process that allows for construction of this representation out of `Mojo` `fn` types.
* Extend the abstract lattice that JAX presents (`dtype` and `shape`) with "richer" information (reflecting underlying tensor information like layouts).

Here's the first concrete goal the author has set for himself:
* Express matmul as a program in the staged DSL, and differentiate it (aim high!)

## Where I'm at:

At `comptime`, able to do the following:

```mojo
fn f(x: TensorLike) -> TensorLike:
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