import os
from pathlib import Path
from typing import Iterable

from lcapy import *
import sympy as s
from sympy.core.evalf import evalf
from sympy.core.numbers import One
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.utilities.codegen import RustCodeGen


def opamp_neg(fb: LaplaceDomainImpedance, gnd: LaplaceDomainImpedance) -> LaplaceDomainImpedance:
    return impedance(1 + fb / gnd)


def create_discrete_statespace(hs: LaplaceDomainImpedance) -> tuple[DTStateSpace, dict[str, ExprDict]]:
    hz = hs.bilinear_transform()
    hzp, hz_defs = hz.parameterize_ZPK()
    return hzp.ss, hz_defs


def statespace_clipper():
    dist = symbol("pdist", real=True, positive=True)
    ff = LSection(C(1e-6), R(10e3)).Vtransfer.as_expr().simplify()
    fb = (R(500e3 * dist) + R(51e3)) | C(51e-9)
    fb = fb.Z.as_expr().simplify()
    fbg = C(0.047e-6) + R(4.7e3)
    fbg = fbg.Z.as_expr().simplify()

    hs = (ff * opamp_neg(fb, fbg)).simplify()
    return create_discrete_statespace(hs)


def tone_h_bass():
    bass_pre = LSection(R(1e3), C(0.22e-6)).chain(Shunt(C(0.22e-6) + R(220))).chain(Shunt(R(10e3) + V(4.5)))
    return bass_pre.Vtransfer.as_expr() * opamp_neg(impedance(1e3), impedance(1)).as_expr()


def tone_h_treble():
    treble_pre = LSection(R(1e3), C(0.22e-6)).chain(Shunt(R(10e3) + V(4.5)))
    treble_gnd = C(0.22e-6) + R(220)
    return treble_pre.Vtransfer.as_expr() * opamp_neg(impedance(1e3), treble_gnd.Z.as_expr()).as_expr()


def lerp(a, b, t):
    return a + (b - a) * t


def statespace_tone():
    tone = symbol("ptone", real=True, positive=True)
    hs = lerp(tone_h_treble().as_expr(), tone_h_bass().as_expr(), tone)
    return create_discrete_statespace(hs)

class MyRustPrinter(RustCodePrinter):
    def _print_Exp1(self, expr, _type=False):
        return "T::simd_e()"

    def _print_Pi(self, expr, _type=False):
        return "T::simd_pi()"

    def _print_Float(self, expr, _type=False):
        ret = str(evalf(expr, expr._prec, {}))
        return f"T::from_f64({ret})"

    def _print_Rational(self, expr):
        if expr.p == 1:
            return f"T::from_f64({self._print(expr.q)}).simd_recip()"

        p, q = tuple(f"T::from_f64({self._print(i)}.0)" for i in (expr.p, expr.q))
        return f"{p} / {q}"

    def _print_MatrixBase(self, A):
        values = ", ".join(self._print(x) for x in A)
        return f"SMatrix::<_, {A.rows}, {A.cols}>::new({values})"

def write_codegen(prefix: str, name: str, state_space: DTStateSpace, params: dict):
    codegen = RustCodeGen(project="ts404", printer=MyRustPrinter())
    routines = [
        codegen.routine(f"{name}_params", [v.sympy for v in params.values()], argument_sequence=None,
                        global_vars=[]),
        codegen.routine(name, tuple(m.sympy for m in [state_space.A, state_space.B, state_space.C, state_space.D]),
                        argument_sequence=None,
                        global_vars=[])
    ]
    codegen.write(routines, str(prefix), to_files=True, header=True)


def codegen_function(name: str, *exprs: s.Expr, public="pub(crate)") -> Iterable[str]:
    from sympy.codegen import Assignment

    printer = RustCodePrinter()
    e = s.Tuple(*exprs)
    sub, simpl = s.cse(e)
    args = ", ".join(f"{name}: T" for name in e.atoms(s.Idx))
    yield f"{public} fn {name}<T: Scalar>({args}) {{"
    for var, e in sub:
        yield f"  let {printer.doprint(Assignment(var, e))}"
    yield "  " + printer.doprint(simpl)
    yield "}}"

if __name__ == "__main__":
    OUT_DIR = "src/gen"
    os.makedirs(OUT_DIR, exist_ok=True)
    write_codegen("src/gen/clipper", "clipper", *statespace_clipper())
    write_codegen("src/gen/tone", "tone", *statespace_tone())
