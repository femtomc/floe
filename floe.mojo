from utils import Variant
from collections import Dict
from memory import ArcPointer
from max.tensor import Tensor


@value
struct ArrayLike:
    var v: Variant[Float32, ExprTracer]

    fn aval(self) -> AbstractValue:
        if self.v.isa[ExprTracer]():
            return aval(self.v[ExprTracer])
        else:
            return aval(self.v[Float32])

    fn val(self) -> Variant[Float32, ExprTracer]:
        return self.v

    def __add__(self: ArrayLike, v: ArrayLike) -> ArrayLike:
        return bind(
            Add,
            List[ArrayLike](self, v),
        )

    def __mul__(self: ArrayLike, v: ArrayLike) -> ArrayLike:
        return bind(
            Mul,
            List[ArrayLike](self, v),
        )


trait Primitive(Writable):
    fn abs(
        self,
        args: List[AbstractValue],
    ) -> AbstractValue:
        ...


fn bind(
    prim: PrimSet,
    args: List[ArrayLike],
) raises -> ArrayLike:
    var interpreter = maybe_find_interpreter(args)
    if interpreter:
        return interpreter.value().interpret(prim, args)
    raise Error()


@value
struct _Add(Primitive):
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("add")

    fn abs(
        self,
        avals: List[AbstractValue],
    ) -> AbstractValue:
        x = avals[0]
        return x


@value
struct _Mul(Primitive):
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("mul")

    fn abs(
        self,
        avals: List[AbstractValue],
    ) -> AbstractValue:
        x = avals[0]
        return x


alias _Primitives = Variant[
    _Add,
    _Mul,
]


@value
struct PrimSet(Primitive):
    var prim: _Primitives

    fn abs(
        self,
        avals: List[AbstractValue],
    ) -> AbstractValue:
        if self.prim.isa[_Add]():
            return self.prim[_Add].abs(avals)
        else:
            return self.prim[_Mul].abs(avals)

    fn write_to[W: Writer](self, mut writer: W):
        if self.prim.isa[_Add]():
            self.prim[_Add].write_to[W](writer)
        else:
            self.prim[_Mul].write_to[W](writer)


alias Add = PrimSet(_Add())
alias Mul = PrimSet(_Mul())


@value
struct Var(EqualityComparable, Hashable, Writable):
    var id: Int
    var aval: AbstractValue

    fn __hash__(self) -> UInt:
        return self.id.__hash__()

    fn __eq__(self: Var, other: Var) -> Bool:
        return self.id == other.id

    fn __ne__(self, other: Var) -> Bool:
        return self.id != other.id

    fn repr(self) -> String:
        return String("%", self.id)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("%", self.id, ":", self.aval)


@value
struct Eqn(Writable):
    var prim: PrimSet
    var invars: List[Var]
    var outvar: Var

    fn write_to[W: Writer](self, mut writer: W):
        self.outvar.write_to(writer)
        writer.write(" = ")
        self.prim.write_to(writer)
        for v in self.invars:
            writer.write(" ")
            writer.write(v[].repr())


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
        writer.write("\n  return ", self.return_var.repr(), " }")


fn dtype_str(dtype: DType) -> String:
    if dtype == DType.float32:
        return "f32"
    else:
        return String(dtype)


@value
struct AbstractValue(Writable):
    var dtype: DType
    var shape: Variant[Tuple[], Tuple[Int]]

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(dtype_str(self.dtype))


@value
struct ExprTracer(EqualityComparable, Hashable, Writable):
    var aval: AbstractValue
    var interpreter: StagingInterpreter
    var id: Int

    fn __hash__(self) -> UInt:
        return self.id.__hash__()

    fn __eq__(self: ExprTracer, other: ExprTracer) -> Bool:
        return self.id == other.id

    fn __ne__(self, other: ExprTracer) -> Bool:
        return self.id != other.id

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("<ExprTracer", self.aval, ">")


fn id(tracer: ExprTracer) -> Int:
    return tracer.id


@value
struct ExprState:
    var equations: List[Eqn]
    var var_counter: Int
    var tracer_to_var: Dict[Int, Var]


fn aval(x: Float32) -> AbstractValue:
    return AbstractValue(DType.float32, ())


fn aval(x: ExprTracer) -> AbstractValue:
    return x.aval


fn aval(x: ArrayLike) -> AbstractValue:
    return x.aval()


fn aval(xs: List[ArrayLike]) -> List[AbstractValue]:
    vs = List[AbstractValue]()
    for x in xs:
        vs.append(aval(x[]))
    return vs


@value
struct StagingInterpreter:
    var ptr: ArcPointer[ExprState]

    fn lift(self, aval: AbstractValue) -> ExprTracer:
        expr_tracer = ExprTracer(
            aval,
            self,
            self.ptr[].var_counter,
        )
        self.ptr[].tracer_to_var[id(expr_tracer)] = Var(
            self.ptr[].var_counter, aval
        )
        self.ptr[].var_counter += 1
        return expr_tracer

    fn lift(mut self, v: Float32) -> ExprTracer:
        return self.pure(v)

    fn wrap(self, v: Float32) -> ArrayLike:
        var expr_tracer = self.pure(v)
        _ = self.add_var(expr_tracer)
        return ArrayLike(expr_tracer)

    fn wrap(self, v: ExprTracer) -> ArrayLike:
        return ArrayLike(v)

    fn wrap(self, v: ArrayLike) -> ArrayLike:
        var val = v.val()
        if val.isa[Float32]():
            return self.wrap(val[Float32])
        else:
            return self.wrap(val[ExprTracer])

    fn add_var(self, tracer: ExprTracer) -> Var:
        var v = Var(id(tracer), tracer.aval)
        self.ptr[].tracer_to_var[id(tracer)] = v
        return v

    fn get_var(self, tracer: ExprTracer) raises -> Var:
        return self.ptr[].tracer_to_var[id(tracer)]

    fn get_var(self, tracers: List[ExprTracer]) raises -> List[Var]:
        var vs = List[Var]()
        for tracer in tracers:
            vs.append(self.get_var(tracer[]))
        return vs

    fn get_var(self, array_like: ArrayLike) raises -> Var:
        var val = array_like.val()
        return self.get_var(val[ExprTracer])

    fn get_var(self, *tracers: ArrayLike) raises -> List[Var]:
        vs = List[Var]()
        for tracer in tracers:
            vs.append(self.get_var(tracer[]))
        return vs

    fn pure(self, v: ExprTracer) -> ExprTracer:
        return v

    fn pure(self, v: Float32) -> ExprTracer:
        return self.lift(aval(v))

    fn pure(self, v: ArrayLike) -> ExprTracer:
        var val = v.val()
        if val.isa[Float32]():
            return self.pure(val[Float32])
        else:
            return self.pure(val[ExprTracer])

    fn pure(self, xs: List[ArrayLike]) -> List[ExprTracer]:
        vs = List[ExprTracer]()
        for x in xs:
            vs.append(self.pure(x[]))
        return vs

    fn interpret(
        mut self,
        prim: PrimSet,
        tracers: List[ArrayLike],
    ) raises -> ArrayLike:
        expr_tracers = self.pure(tracers)
        avals_in = aval(tracers)
        aval_out = prim.abs(avals_in)
        out_tracer = self.lift(aval_out)
        invars = self.get_var(expr_tracers)
        outvar = self.add_var(out_tracer)
        self.ptr[].equations.append(Eqn(prim, invars, outvar))
        return ArrayLike(out_tracer)


fn staging_interpreter() -> StagingInterpreter:
    ptr = ExprState(
        List[Eqn](),
        0,
        Dict[Int, Var](),
    )
    return StagingInterpreter(ptr)


alias Interpreter = StagingInterpreter


fn maybe_find_interpreter(
    vs: List[ArrayLike],
) -> Optional[Interpreter]:
    for v in vs:
        var val = v[].val()
        if val.isa[ExprTracer]():
            return val[ExprTracer].interpreter
    return None


fn stage1[
    f: fn (ArrayLike) raises -> ArrayLike
](x: ArrayLike,) raises -> Expr:
    interp = staging_interpreter()
    v = interp.wrap(x)
    out = f(v)
    var expr = Expr(
        List[Var](interp.get_var(v)),
        interp.ptr[].equations,
        interp.get_var(out),
    )
    return expr


fn array(v: Float32) -> ArrayLike:
    return ArrayLike(v)


def main():
    def f(x: ArrayLike) -> ArrayLike:
        return x + x + x * x

    var expr = stage1[f](array(3.0))
    print(expr)
