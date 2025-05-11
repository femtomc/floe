from layout import LayoutTensor, Layout
from layout.int_tuple import IntTuple
from layout.math import sum
from memory import UnsafePointer

# Little named tensor sketching


@value
struct f32[names: List[String], layout: Layout](Writable):
    var val: LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
    ]

    @staticmethod
    fn ones() -> f32[names, layout]:
        return f32[names](
            LayoutTensor[DType.float32, layout, MutableAnyOrigin](
                UnsafePointer[Scalar[DType.float32], alignment=64].alloc(
                    layout.size()
                )
            ).fill(1.0)
        )

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.val)

    fn sum[
        name: String
    ](self) -> f32[remove(names, name), layout[findindex(names, name, 0)]]:
        alias new_layout = layout[findindex(names, name, 0)]
        var out_tens = LayoutTensor[
            DType.float32, new_layout, MutableAnyOrigin
        ](
            UnsafePointer[Scalar[DType.float32], alignment=64].alloc(
                new_layout.size()
            )
        )
        var new_tens = sum[findindex(names, name, 0)](self.val, out_tens)
        return f32[remove(names, name)](out_tens)


fn findindex[
    v: Copyable & Movable & EqualityComparable
](l: List[v], idx: v, fallback: Int) -> Int:
    for _idx in range(len(l)):
        if idx == l[_idx]:
            return _idx
    return fallback


fn remove[
    v: Copyable & Movable & EqualityComparable
](l: List[v], idx: v) -> List[v]:
    var out = List[v]()
    for _idx in range(len(l)):
        if l[_idx] == idx:
            pass
        else:
            out.append(l[_idx])
    return out


fn main():
    var v = f32[
        names = List[String]("x", "y"),
        layout = Layout.col_major(20, 20),
    ].ones().sum["x"]()
    print(v)
