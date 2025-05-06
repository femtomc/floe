from utils import Variant
from collections import Dict
from memory import OwnedPointer, UnsafePointer
from max.tensor import Tensor

alias Values = Variant[Float32,]

alias Atom = Variant[
    Values,
    ExprTracer,
]


fn maybe_find_interpreter(
    v: VariadicListMem[Atom],
) -> Optional[Interpreter]:
    pass


fn unwrap(v: Float32) -> Float32:
    return v


fn unwrap(v: ExprTracer) -> AbstractValue:
    return v.tracer


fn unwrap(
    vs: VariadicListMem[Atom],
) -> VariadicListMem[Atom]:
    new = VariadicListMem[Atom]()
    for v in vs:
        new.append(unwrap(v))
    return new


trait Primitive(Writable):
    def impl(self, args: VariadicList[Float32]) -> Float32:
        ...

    def bind(self, args: VariadicListMem[Atom]) -> Atom:
        ...


fn bind[
    P: Primitive
](prim: P, args: VariadicListMem[Atom],) -> Atom:
    var interpreter = maybe_find_interpreter(args)
    if interpreter:
        return interpreter.value().interpret(_Add, args)
    else:
        return prim.impl(args)


@value
struct _Add(Primitive):
    var name: String

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.name)

    def bind(self, args: VariadicListMem[Atom]) -> Atom:
        return bind(self, args)

    def impl(self, args: VariadicList[Float32]) -> Float32:
        x = args[0]
        y = args[1]
        return x + y


alias _Primitives = Variant[_Add,]


@value
struct PrimSet(Primitive):
    var prim: _Primitives

    def impl(self, args: VariadicList[Float32]) -> Float32:
        if self.prim.isa[_Add]():
            return self.prim[_Add].impl(args)

    def bind(self, args: VariadicListMem[Atom]) -> Atom:
        if self.prim.isa[_Add]():
            return self.prim[_Add].bind(args)

    fn write_to[W: Writer](self, mut writer: W):
        if self.prim.isa[_Add]():
            self.prim[_Add].write_to[W](writer)


@value
struct Var(EqualityComparable, Hashable, Writable):
    var name: String

    fn __hash__(self) -> UInt:
        return self.name.__hash__()

    fn __eq__(self: Var, other: Var) -> Bool:
        return self.name == other.name

    fn __ne__(self, other: Var) -> Bool:
        return self.name != other.name

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("%", self.name)


@value
struct Eqn(Writable):
    var invars: List[Var]
    var outvar: Var
    var prim: PrimSet

    fn write_to[W: Writer](self, mut writer: W):
        self.outvar.write_to(writer)
        writer.write(" = ")
        self.prim.write_to(writer)
        for v in self.invars:
            writer.write(" ")
            v[].write_to(writer)


@value
struct Expr(Writable):
    var parameters: List[Var]
    var equations: List[Eqn]
    var return_var: Var

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("{ lambda")
        for v in self.parameters:
            writer.write(" ", v[], " ")
        writer.write(".")
        for eqn in self.equations:
            writer.write("\n")
            writer.write("  ", eqn[])
        writer.write("\n  return ", self.return_var, " }")


@value
struct AbstractValue:
    var dtype: DType
    var shape: List[Int]


@value
struct ExprTracer:
    var aval: AbstractValue
    var interpreter: UnsafePointer[StagingInterpreter]


@value
struct EvalTracer:
    var v: Values


@value
struct StagingInterpreter:
    var equations: List[Eqn]
    var name_counter: Int

    fn fresh_var(mut self) -> Var:
        self.name_counter += 1
        return Var(String(self.name_counter))

    fn interpret(mut self, prim: Primitive, args: List[Var]) -> Var:
        binder = self.fresh_var()
        self.equations.append(Eqn(args, binder, prim))
        return binder


def make_expr[f: fn (List[Var]) raises -> Var, num_args: Int]() -> Expr:
    interpreter = StagingInterpreter(List[Eqn](), 0)
    parameters = List[Var]()
    for _ in range(num_args):
        parameters.append(interpreter.fresh_var())
    stack.append(interpreter)
    var result = f(parameters)
    var final = stack.pop()
    return Expr(parameters, final.equations, result)


@value
struct EvalInterpreter:
    var env: Dict[Var, Float32]

    def eval(self, prim: Primitive, vars: List[Var]) -> Float32:
        var v1 = self.env[vars[0]]
        var v2 = self.env[vars[1]]
        return v1 + v2


fn write(mut env: Dict[Var, Float32], v: Var, val: Float32):
    env[v] = val


def read(env: Dict[Var, Float32], v: Var) -> Float32:
    return env[v]


def eval_expr[expr: Expr](binders: List[Float32]) -> Float32:
    var init_env = Dict[Var, Float32]()
    for idx in range(len(expr.parameters)):
        var v = expr.parameters[idx]
        var val = binders[idx]
        init_env[v] = val
    interp = EvalInterpreter(init_env)
    for eqn in expr.equations:
        var outval = interp.eval(eqn[].prim, eqn[].invars)
        write(interp.env, eqn[].outvar, outval)
    return read(interp.env, expr.return_var)


def main():
    def f(_x: List[Var]) -> Var:
        x = _x[0]
        v = add(x, x)
        return add(v, v)

    var expr = make_expr[f, 1]()
