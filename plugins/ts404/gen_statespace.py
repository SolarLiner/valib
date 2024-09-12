from pathlib import Path
from typing import Iterable

import sympy as sm
from lcapy import *
from sympy.printing.rust import RustCodePrinter
from sympy.utilities.codegen import RustCodeGen


def opamp_noninverting(zf, zg):
    return 1 + zf / zg


def create_discrete_statespace(hs: LaplaceDomainImpedance) -> DTStateSpace:
    hz = hs.bilinear_transform().simplify()
    return hz.ss


def statespace_input():
    pre = LSection(C(0.02e-6) + R(1e3), R(510e3) + V(4.5))
    post = Shunt(R(10e3)).chain(Series(C(1e-6)))
    h = (pre.Vtransfer.as_expr() * post.Vtransfer.as_expr()).simplify()
    return create_discrete_statespace(h)


def statespace_clipper():
    dist = symbol("pdist", real=True, positive=True)
    ff = LSection(C(1e-6), R(10e3)).Vtransfer.as_expr().simplify()
    fb = (R(500e3 * dist) + R(51e3)) | C(51e-9)
    fb = fb.Z.as_expr().simplify()
    fbg = C(0.047e-6) + R(4.7e3)
    fbg = fbg.Z.as_expr().simplify()

    # Explicitely applying gain (even through it should have been implicit in the complex impedances??)
    hs = (ff * opamp_noninverting(fb, fbg)).simplify() * lerp(12, 118, dist)
    return create_discrete_statespace(hs)


def tone_h_bass():
    bass_pre = LSection(R(1e3), C(0.22e-6)).chain(Shunt(C(0.22e-6) + R(220))).chain(Shunt(R(10e3) + V(4.5)))
    return bass_pre.Vtransfer.as_expr() * opamp_noninverting(1e3, sm.oo).as_expr()


def tone_h_treble():
    treble_pre = LSection(R(1e3), C(0.22e-6)).chain(Shunt(R(10e3) + V(4.5)))
    treble_gnd = C(0.22e-6) + R(220)
    return treble_pre.Vtransfer.as_expr() * opamp_noninverting(impedance(1e3), treble_gnd.Z.as_expr()).as_expr()


def lerp(a, b, t):
    return a + (b - a) * t


def g_taper(x):
    x = 2 * x - 1
    y = lerp(x, x ** 3, 0.75)
    return (y + 1) / 2


def statespace_tone():
    tone = symbol("ptone", real=True, positive=True)
    hs = lerp(tone_h_treble().as_expr(), tone_h_bass().as_expr(),g_taper(tone))
    return create_discrete_statespace(hs)


def statespace_output():
    pre = LSection(C(0.1e-6), R(510e3) + V(4.5))
    post = Shunt(R(10e3)).chain(Series(R(100) + C(1e-6))).chain(Shunt(R(10e3)))
    h = (pre.Vtransfer.as_expr() * post.Vtransfer.as_expr()).simplify()
    return create_discrete_statespace(h)


class MyRustPrinter(RustCodePrinter):
    def __init__(self):
        RustCodePrinter.__init__(self, {'user_functions': {'recip': 'simd_recip'}})

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


def write_codegen(prefix: str, name: str, state_space: DTStateSpace, params: dict):
    codegen = RustCodeGen(project="ts404", printer=MyRustPrinter())
    routines = [
        codegen.routine(f"{name}_params", [v.sympy for v in params.values()], argument_sequence=None, global_vars=[]),
        codegen.routine(name, tuple(m.sympy for m in [state_space.A, state_space.B, state_space.C, state_space.D]),
                        argument_sequence=None, global_vars=[])]
    codegen.write(routines, str(prefix), to_files=True, header=True)


def codegen_header() -> Iterable[str]:
    yield from ["#![allow(unused)]", "#![allow(non_snake_case)]", "", "use nalgebra::SMatrix;", "use valib::Scalar;",
                "use valib::filters::statespace::StateSpace;", ]


def codegen_statespace(name: str, state_space: DTStateSpace, public="pub(crate)") -> Iterable[str]:
    from sympy.codegen import Assignment

    def postprocess_codegen(s: str) -> str:
        # Replacing in string after printing because there's no easy way of doing it from the printer
        return s.replace('.recip(', '.simd_recip(').replace('.powi(', '.simd_powf(').replace('.powf(', '.simd_powf(')

    nin = state_space.Nu
    nstate = state_space.Nx
    nout = state_space.Ny

    printer = MyRustPrinter()
    e = sm.Tuple(state_space.A, state_space.B, state_space.C, state_space.D)
    sub, simpl = sm.cse(e)
    args = ", ".join(f"{name}: T" for name in sorted(e.atoms(sm.Symbol), key=lambda v: str(v)))
    yield f"{public} fn {name}<T: Scalar>({args}) -> StateSpace<T, {nin}, {nstate}, {nout}> {{"
    for var, e in sub:
        # Replacing in string after printing because there's no easy way of doing it from the printer
        pp = printer.doprint(Assignment(var, e))
        yield f"  let {postprocess_codegen(pp)};"
    yield "  StateSpace::new("
    for e in simpl[0]:
        pp = printer.doprint(e)
        yield f"    {postprocess_codegen(pp)},"
    yield "  )"
    yield "}"


def codegen_full(*names: tuple[str, DTStateSpace], public="pub(crate)") -> Iterable[str]:
    yield from codegen_header()
    yield ""
    for name, state_space in names:
        yield from codegen_statespace(name, state_space, public)
        yield ""


if __name__ == "__main__":
    OUT_FILE = Path("src/gen.rs")
    OUT_FILE.write_text("\n".join(codegen_full(("input", statespace_input()), ("clipper", statespace_clipper()),
                                               ("tone", statespace_tone()), ("output", statespace_output()))))
