from collections.abc import Iterable
from dataclasses import dataclass

import lcapy
import sympy
from sympy.codegen import Assignment
from sympy.printing.rust import RustCodePrinter

from valib.codegen import Function, Visibility, Generatable, ValibPrinter, SourceFile


@dataclass
class StateSpace(Generatable):
    obj: lcapy.DTStateSpace

    def as_function(self, name: str, visibility=Visibility.PRIVATE) -> tuple[set[str], Function]:
        atoms = [atom for atom in sympy.Tuple(self.obj.A, self.obj.B, self.obj.C, self.obj.D).atoms(sympy.Symbol)]
        atoms.sort(key=str)
        nin = self.obj.Nu
        nstate = self.obj.Nx
        nout = self.obj.Ny
        func = Function(name, [(str(a), "T") for a in atoms],
                        [("T", "Scalar")], self, f"StateSpace<T, {nin}, {nstate}, {nout}>",
                        visibility)
        return {"nalgebra::SMatrix", "valib::Scalar", "valib::filters::statespace::StateSpace"}, func

    def as_source_file(self, function_name: str, function_visibility=Visibility.PRIVATE) -> SourceFile:
        uses, func = self.as_function(function_name, function_visibility)
        return SourceFile(uses=uses, functions={function_name: func})

    def __call__(self, printer: RustCodePrinter = ValibPrinter()) -> Iterable[str]:
        nin = self.obj.Nu
        nstate = self.obj.Nx
        nout = self.obj.Ny
        vars, exprs = sympy.cse((self.obj.A, self.obj.B, self.obj.C, self.obj.D))
        for var, expr in vars:
            yield printer.doprint(Assignment(var, expr))

        yield f"StateSpace::<_, {nin}, {nstate}, {nout}>::new("
        for expr in exprs:
            yield f"    {printer.doprint(expr)},"
        yield ")"
