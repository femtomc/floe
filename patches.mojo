from utils import Variant
from collections import Dict
from memory import ArcPointer, UnsafePointer
from layout import Layout, LayoutTensor


@value
struct STensor:
    var aval: AbstractValue


alias _Tens = Variant[STensor, ExprTracer]


@value
struct Tensor:
    var v: _Tens

    fn aval(self) -> AbstractValue:
        if self.v.isa[ExprTracer]():
            return aval(self.v[ExprTracer])
        else:
            return aval(self.v[STensor])

    fn val(self) -> _Tens:
        return self.v

    fn __add__(self: Tensor, v: Tensor) raises -> Tensor:
        return bind(
            Add,
            List[Tensor](self, v),
        )

    fn __mul__(self: Tensor, v: Tensor) raises -> Tensor:
        return bind(
            Mul,
            List[Tensor](self, v),
        )


trait Primitive(Writable):
    fn abs(
        self,
        args: List[AbstractValue],
    ) -> AbstractValue:
        ...


fn bind(
    prim: PrimSet,
    args: List[Tensor],
) raises -> Tensor:
    var interpreter = maybe_find_interpreter(args)
    return interpreter.value().interpret(prim, args)


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
    var layout: Layout

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(dtype_str(self.dtype))
        writer.write("[", self.layout, "]")


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


fn aval(x: STensor) -> AbstractValue:
    return x.aval


fn aval[
    dtype: DType, layout: Layout
](x: LayoutTensor[dtype, layout]) -> AbstractValue:
    return AbstractValue(dtype, layout)


fn aval(x: ExprTracer) -> AbstractValue:
    return x.aval


fn aval(x: Tensor) -> AbstractValue:
    return x.aval()


fn aval(xs: List[Tensor]) -> List[AbstractValue]:
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

    fn lift(mut self, v: STensor) -> ExprTracer:
        return self.pure(v)

    fn wrap(self, v: STensor) -> Tensor:
        var expr_tracer = self.pure(v)
        _ = self.add_var(expr_tracer)
        return Tensor(expr_tracer)

    fn wrap(self, v: ExprTracer) -> Tensor:
        return Tensor(v)

    fn wrap(self, v: Tensor) -> Tensor:
        var val = v.val()
        if val.isa[STensor]():
            return self.wrap(val[STensor])
        else:
            return self.wrap(val[ExprTracer])

    fn add_var(self, tracer: ExprTracer) -> Var:
        var v = Var(id(tracer), tracer.aval)
        self.ptr[].tracer_to_var[id(tracer)] = v
        return v

    fn get_var(self, tracer: ExprTracer) -> Optional[Var]:
        return self.ptr[].tracer_to_var.find(id(tracer))

    fn get_var(self, tracers: List[ExprTracer]) -> Optional[List[Var]]:
        var vs = List[Var]()
        for tracer in tracers:
            var v = self.get_var(tracer[])
            if v:
                vs.append(v.value())
            else:
                return None
        return Optional(vs)

    fn get_var(self, array_like: Tensor) -> Optional[Var]:
        var val = array_like.val()
        return self.get_var(val[ExprTracer])

    fn get_var(self, *tracers: Tensor) -> Optional[List[Var]]:
        vs = List[Var]()
        for tracer in tracers:
            var v = self.get_var(tracer[])
            if v:
                vs.append(v.value())
            else:
                return None
        return Optional(vs)

    fn pure(self, v: ExprTracer) -> ExprTracer:
        return v

    fn pure(self, v: STensor) -> ExprTracer:
        return self.lift(aval(v))

    fn pure(self, v: Tensor) -> ExprTracer:
        var val = v.val()
        if val.isa[STensor]():
            return self.pure(val[STensor])
        else:
            return self.pure(val[ExprTracer])

    fn pure(self, xs: List[Tensor]) -> List[ExprTracer]:
        vs = List[ExprTracer]()
        for x in xs:
            vs.append(self.pure(x[]))
        return vs

    fn interpret(
        mut self,
        prim: PrimSet,
        tracers: List[Tensor],
    ) raises -> Tensor:
        expr_tracers = self.pure(tracers)
        avals_in = aval(tracers)
        aval_out = prim.abs(avals_in)
        out_tracer = self.lift(aval_out)
        var invars = self.get_var(expr_tracers)
        if invars:
            outvar = self.add_var(out_tracer)
            self.ptr[].equations.append(Eqn(prim, invars.value(), outvar))
            return Tensor(out_tracer)
        else:
            raise Error()


fn staging_interpreter() -> StagingInterpreter:
    ptr = ExprState(
        List[Eqn](),
        0,
        Dict[Int, Var](),
    )
    return StagingInterpreter(ptr)


alias Interpreter = StagingInterpreter


fn maybe_find_interpreter(
    vs: List[Tensor],
) -> Optional[Interpreter]:
    for v in vs:
        var val = v[].val()
        if val.isa[ExprTracer]():
            return val[ExprTracer].interpreter
    return None


fn stage1[
    f: fn (Tensor) raises -> Tensor
](x: LayoutTensor,) -> Optional[Expr]:
    interp = staging_interpreter()
    var x_ = STensor(AbstractValue(x.dtype, x.layout))
    v = interp.wrap(x_)
    try:
        out = f(v)
        var inval = interp.get_var(v)
        var outvar = interp.get_var(out)
        var expr = Expr(
            List[Var](inval.value()),
            interp.ptr[].equations,
            outvar.value(),
        )
        return Optional(expr)
    except:
        return None


fn tensor[dtype: DType, layout: Layout]() -> Tensor:
    alias absval = AbstractValue(dtype, layout)
    return Tensor(STensor(absval))


fn ones[
    dtype: DType,
    layout: Layout,
]() -> LayoutTensor[dtype, layout, MutableAnyOrigin]:
    return LayoutTensor[dtype, layout, MutableAnyOrigin](
        UnsafePointer[Scalar[dtype], alignment=64].alloc(layout.size())
    ).fill(1.0)


def main():
    def f(x: Tensor) -> Tensor:
        def g(x: Tensor) -> Tensor:
            return x * x

        return g(x + x + x * x)

    alias expr = stage1[f](
        ones[
            DType.float32,
            Layout.col_major(3, 4),
        ]()
    )
    print(expr.value())
