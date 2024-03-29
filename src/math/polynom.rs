use std::{iter, ops};

use num_traits::real::Real;
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct Polynom<T>(pub Vec<T>);

impl<T> FromIterator<T> for Polynom<T> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<T> Polynom<T> {
    pub fn polyline(offset: T, scale: T) -> Self {
        Self(vec![offset, scale])
    }

    pub fn map<U>(self, map: impl FnMut(T) -> U) -> Polynom<U> {
        Polynom(self.0.into_iter().map(map).collect())
    }
}

impl<T: Zero + PartialEq> PartialEq for Polynom<T> {
    fn eq(&self, other: &Self) -> bool {
        self.degree() == other.degree() && self.0[..self.degree()] == other.0[..other.degree()]
    }
}

impl<T: Zero + Eq> Eq for Polynom<T> {}

impl<T: Zero> Polynom<T> {
    pub fn degree(&self) -> usize {
        self.0.iter().rposition(|s| !s.is_zero()).unwrap_or(0)
    }

    pub fn canonicalize_in_place(&mut self) {
        for _ in self.0.drain(self.degree() + 1..) {}
    }

    pub fn canonicalize(mut self) -> Self {
        self.canonicalize_in_place();
        self
    }

    pub fn get(&self, rank: usize) -> T
    where
        T: Copy,
    {
        if rank <= self.degree() {
            self.0[rank]
        } else {
            T::zero()
        }
    }
}

impl<T: Real> Polynom<T> {
    pub fn eval(&self, x: T) -> T {
        self.0
            .iter()
            .copied()
            .enumerate()
            .fold(T::zero(), |acc, (i, s)| acc + s * x.powi(i as _))
    }
}

impl<T> ops::Index<usize> for Polynom<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> ops::IndexMut<usize> for Polynom<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: ops::Neg> ops::Neg for Polynom<T> {
    type Output = Polynom<T::Output>;

    fn neg(self) -> Self::Output {
        Polynom(self.0.into_iter().map(T::neg).collect())
    }
}

impl<T: Copy + ops::Add<T>> ops::Add<Self> for Polynom<T> {
    type Output = Polynom<T::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        Polynom(
            self.0
                .iter()
                .copied()
                .zip(rhs.0.iter().copied())
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

impl<T: Copy + ops::Sub<T>> ops::Sub<Self> for Polynom<T> {
    type Output = Polynom<T::Output>;

    fn sub(self, rhs: Self) -> Self::Output {
        Polynom(
            self.0
                .iter()
                .copied()
                .zip(rhs.0.iter().copied())
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

impl<T: Copy + Zero + ops::Add<T>> iter::Sum<Polynom<T::Output>> for Polynom<T> {
    fn sum<I: Iterator<Item = Polynom<T::Output>>>(iter: I) -> Self {
        iter.fold(Self::zero(), ops::Add::add)
    }
}

impl<T: Copy + Zero + One + PartialEq + ops::AddAssign<T> + ops::Mul<T, Output = T>> ops::Mul<Self>
    for Polynom<T>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        if self.is_one() {
            return rhs;
        }
        if rhs.is_one() {
            return self;
        }

        let a = &self.0[..=self.degree()];
        let b = &rhs.0[..=rhs.degree()];
        let mut out = vec![T::zero(); a.len() + b.len() - 1];
        convolve(&mut out, a, b);
        Self(out)
    }
}

impl<T: Zero> Zero for Polynom<T>
where
    Self: ops::Add<Self, Output = Self>,
{
    fn zero() -> Self {
        Self(vec![])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|s| s.is_zero())
    }
}

impl<T: Zero + One> One for Polynom<T>
where
    Self: ops::Mul<Self, Output = Self>,
{
    fn one() -> Self {
        Self(vec![T::one()])
    }
}

impl<T: Copy + One + ops::Neg<Output = T>> Polynom<T>
where
    Self: One,
{
    pub fn from_roots(roots: impl IntoIterator<Item = T>) -> Self {
        roots
            .into_iter()
            .map(|r| Self::polyline(-r, T::one()))
            .fold(Self::one(), ops::Mul::mul)
    }
}

// Naive convolution algorithm, suitable to the relatively small arrays ofr polynomial coefficients here
// See the Karatsuba convolution algorithm for a non-FFT algorithm that is better than O(n^2) (but worse than linear time)
fn convolve<T: Copy + Zero + ops::AddAssign<T> + ops::Mul<T, Output = T>>(
    output: &mut [T],
    a: &[T],
    b: &[T],
) {
    let (a, b) = if a.len() < b.len() { (a, b) } else { (b, a) };

    output.fill(T::zero());
    for k in 0..a.len() {
        for i in 0..b.len() {
            output[k + i] += a[k] * b[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};
    use rstest::rstest;

    use crate::math::polynom::{convolve, Polynom};

    #[rstest]
    #[case(0, Polynom::zero())]
    #[case(0, Polynom::one())]
    #[case(2, Polynom(vec ! [0.0, 1.0, 2.0, 0.0]))]
    fn test_degree(#[case] expected: usize, #[case] p: Polynom<f32>) {
        let actual = p.degree();
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(& [3.0, 8.0, 14.0, 8.0, 3.0], & [1.0, 2.0, 3.0], & [3.0, 2.0, 1.0])]
    fn test_convolution(#[case] expected: &[f32], #[case] a: &[f32], #[case] b: &[f32]) {
        let mut actual = expected.to_vec();
        actual.fill(0.0);

        convolve(&mut actual, a, b);

        assert_eq!(expected, &*actual);
    }

    #[rstest]
    #[case(Polynom::zero(), Polynom::one(), Polynom::zero())]
    #[case(Polynom(vec ! [1.0, 2.0]), Polynom(vec ! [1.0, 2.0]), Polynom::one())]
    #[case(Polynom(vec ! [1.0, 2.0]), Polynom::one(), Polynom(vec ! [1.0, 2.0]))]
    #[case(Polynom(vec ! [- 1.0, 1.0, 2.0]), Polynom(vec ! [1.0, 1.0]), Polynom(vec ! [- 1.0, 2.0]))]
    fn test_mul(#[case] expected: Polynom<f32>, #[case] a: Polynom<f32>, #[case] b: Polynom<f32>) {
        let actual = a * b;
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(Polynom(vec ! [1.0, 2.0, 1.0]), vec ! [- 1.0, - 1.0])]
    fn test_polynom_from_roots(#[case] expected: Polynom<f32>, #[case] roots: Vec<f32>) {
        let actual = Polynom::from_roots(roots);
        assert_eq!(expected, actual);
    }
}
