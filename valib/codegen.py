import enum
import os
from dataclasses import dataclass, field
from typing import Protocol, Iterable, Optional

import sympy
from sympy.codegen import Assignment
from sympy.core._print_helpers import Printable
from sympy.printing.rust import RustCodePrinter
from sympy.utilities.codegen import Routine


def postprocess_print(s: str) -> str:
    # Replacing in string after printing because there's no easy way of doing it from the printer
    return s.replace('.recip(', '.simd_recip(').replace('.powi(', '.simd_powf(').replace('.powf(', '.simd_powf(')


class ValibPrinter(RustCodePrinter):
    def __init__(self):
        RustCodePrinter.__init__(self, {'user_functions': {'recip': 'simd_recip'}})

    def doprint(self, expr, assign_to=None):
        ret = super().doprint(expr, assign_to)
        return postprocess_print(ret)

    def _print_Zero(self, expr):
        return "T::from_f64(0f64)"

    def _print_Exp1(self, expr, _type=False):
        return "T::simd_e()"

    def _print_Pi(self, expr, _type=False):
        return "T::simd_pi()"

    def _print_Integer(self, expr, _type=False):
        s = str(expr)
        return f"T::from_f64({s}f64)"

    def _print_Float(self, expr, _type=False):
        ret = str(expr)
        return f"T::from_f64({ret})"

    def _print_Rational(self, expr):
        if expr.p == 1:
            return f"T::from_f64({self._print(expr.q)}).simd_recip()"

        p, q = tuple(f"T::from_f64({self._print(i)}f64)" for i in (expr.p, expr.q))
        return f"{p} / {q}"

    def _print_MatrixBase(self, A):
        values = ", ".join(self._print(x) for x in A)
        return f"SMatrix::<_, {A.rows}, {A.cols}>::new({values})"


class Visibility(enum.StrEnum):
    PRIVATE = ""
    SELF = "pub(self) "
    SUPER = "pub(super) "
    CRATE = "pub(crate) "
    PUBLIC = "pub "


class Generatable(Protocol):
    def __call__(self, printer: RustCodePrinter = ValibPrinter()) -> Iterable[str]: ...

    def print(self, printer: RustCodePrinter = ValibPrinter(), depth: int = 0) -> Iterable[str]:
        leading = "    " * depth
        for line in self(printer):
            yield f"{leading}{line}"

    def __str__(self) -> str:
        return "\n".join(self())

    def write_to(self, path: os.PathLike, printer:RustCodePrinter=ValibPrinter()) -> None:
        with open(path, "w") as f:
            f.writelines((f"{s}\n" for s in self(printer)))


@dataclass
class Function(Generatable):
    name: str
    params: list[tuple[str, str]]
    type_params: list[tuple[str, str]]
    contents: Generatable
    return_type: Optional[str] = None
    visibility: Visibility = Visibility.PRIVATE

    def __call__(self, printer: RustCodePrinter = ValibPrinter()) -> Iterable[str]:
        def trait_bound_repr(s: str) -> str:
            if len(s) == 0:
                return ""
            return f": {s}"

        params = ", ".join((f"{name}: {type}" for name, type in self.params))
        type_params = ", ".join((f"{name}{trait_bound_repr(trait)}" for name, trait in self.type_params))

        if len(type_params) > 0:
            type_params = f"<{type_params}>"

        rettype = f"-> {self.return_type}" if self.return_type is not None else ""

        yield f"{self.visibility}fn {self.name}{type_params}({params}) {rettype} {{"
        yield from self.contents.print(printer, 1)
        yield "}"


@dataclass
class SourceFile(Generatable):
    features: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    functions: dict[str, Function] = field(default_factory=dict)

    def __or__(self, other: "SourceFile") -> "SourceFile":
        return SourceFile(
            features=self.features | other.features,
            uses=self.uses | other.uses,
            functions=self.functions | other.functions,
        )

    def __call__(self, printer: RustCodePrinter = ValibPrinter()) -> Iterable[str]:
        for feature in self.features:
            yield f"#![feature({feature})]"

        for use in self.uses:
            yield f"use {use};"

        yield ""

        for func in self.functions.values():
            yield from func.print(printer)
            yield ""