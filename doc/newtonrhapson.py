
from dataclasses import dataclass, field
from itertools import repeat
from typing import Callable, TypeAlias

import numpy as np
from sympy import Symbol, Expr, Eq, diff, lambdify, simplify

from signalmath import Signal, rms

RootEqu: TypeAlias = Callable[[Signal, Signal], Signal]

@dataclass
class NewtonRhapson:
    """Implementation of the Newton-Rhapson method.
    Resolution is done by iterating up to n times, or when the step modifies the guess
    by less than `tol` RMS."""

    eval: RootEqu
    differential: RootEqu
    n: int = 0
    tol: float = 1e-6
    iter: list[Signal] = field(default_factory=list)

    def __call__(self, x: Signal, initial:Signal|None=None) -> Signal:
        """Evaluate the implicit equation at x, with the initial solution provided (default: 0)."""
        if initial is None:
            initial = np.zeros_like(x)
        guess=np.copy(initial)
        self.iter = []
        it = range(self.n) if self.n > 0 else repeat(0)
        for _ in it:
            dy = self.differential(x, guess)
            if rms(dy) < self.tol:
                break
            step = self.eval(x,guess) / dy
            srms = rms(step)
            if srms < self.tol:
                break
            # tqdm.write(f"{i:3}: {db(srms):2.3} dB RMS")
            guess -= step
            self.iter.append(np.copy(guess))
        # print(f"guessed within {db(srms)} dB")
        return guess


class NRSymbolic(NewtonRhapson):
    """Implementation of the Newton-Rhapson method over a symbolic SymPy expression.
    Resolution is done by iterating up to n times, or when the step modifies the guess
    by less than `tol` RMS."""

    iteration: Expr
    differential: Expr
    inv_differential: Expr

    def __init__(self, x: Symbol, y: Symbol, expr: Expr|Eq, *args, **kwargs):
        if isinstance(expr, Eq):
            expr = expr.rhs - expr.lhs
        self.iteration = expr
        self.differential = diff(self.iteration, y)
        self.inv_differential = simplify(1/self.differential)
        super().__init__(lambdify([x,y], self.iteration, 'scipy'), lambdify([x,y], self.differential, 'scipy'), *args, **kwargs)
