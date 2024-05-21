use nalgebra::{
    Complex, DMatrix, DVectorView, DVectorViewMut, DefaultAllocator, Dim, Dyn, Matrix, OMatrix,
    Storage, VecStorage,
};
use std::{iter, ops};

use crate::math::newton_rhapson_mono;
use crate::Scalar;
use num_traits::real::Real;
use num_traits::{Float, One, Zero};
use simba::scalar::ComplexField;

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

impl<T: Scalar> Polynom<T> {
    pub fn eval(&self, x: T) -> T {
        // Implementation using Horner's rule
        self.0
            .iter()
            .rev()
            .copied()
            .reduce(|a, b| a + x * b)
            .unwrap_or_else(T::zero)
    }

    pub fn diff(&self) -> Self {
        let data = self
            .0
            .iter()
            .copied()
            .enumerate()
            .skip(1)
            .map(|(i, x)| T::from_f64(i as _) * x)
            .collect();
        Self(data)
    }
}

impl<T: Scalar + nalgebra::RealField + nalgebra::ComplexField> Polynom<T>
where
    DefaultAllocator: nalgebra::allocator::Allocator<T, Dyn, Dyn>,
{
    pub fn companion_matrix(&self) -> DMatrix<T> {
        let rank = self.0.len();
        let dyn_rank = Dyn(rank);
        let mut ret = DMatrix::from_data(VecStorage::new(
            dyn_rank,
            dyn_rank,
            iter::repeat_with(T::zero).take(rank * rank).collect(),
        ));
        ret.view_mut((1, 0), (rank - 1, rank - 1))
            .fill_with_identity();
        //ret.column_mut(rank - 1) = DVectorViewMut::from(self.0.as_mut_slice());
        for (i, x) in ret.column_mut(rank - 1).iter_mut().enumerate() {
            *x = -self.0[i];
        }
        ret
    }

    pub fn roots(&self) -> Vec<T> {
        if self.degree() == 0 {
            return vec![];
        }

        let mut p = self.clone();
        let mut zeros = vec![];
        let cast = <T as Scalar>::from_f64;
        while p.degree() > 1 {
            let pdiff = p.diff();
            let z = newton_rhapson_mono(
                |x| p.eval(x),
                |x| pdiff.eval(x),
                cast(1e-6),
                100,
                cast(100.0),
            );
            zeros.push(z);
            p /= Self::polyline(-z, T::one());
        }

        // Final zero found by solving the linear equation p[1] * x + p[0] = 0
        let z = -p.0[0] / p.0[1];
        zeros.push(z);
        zeros
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

impl<T: Copy + ops::AddAssign<T>> ops::AddAssign<Self> for Polynom<T> {
    fn add_assign(&mut self, rhs: Self) {
        for (y, x) in self.0.iter_mut().zip(rhs.0.iter().copied()) {
            *y += x;
        }
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

impl<T: Copy + ops::SubAssign<T>> ops::SubAssign<Self> for Polynom<T> {
    fn sub_assign(&mut self, rhs: Self) {
        for (y, x) in self.0.iter_mut().zip(rhs.0.iter().copied()) {
            *y -= x;
        }
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
    use nalgebra::{Complex, DMatrix, Dyn, VecStorage};
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

    #[test]
    fn test_diff() {
        let p = Polynom::polyline(1f64, 3.0);
        let actual = p.diff();
        let expected = Polynom::from_iter([3f64]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_companion_matrix() {
        let p = Polynom::<f64>::from_iter([1.0, 2.0, 3.0, 4.0]);
        let actual = p.companion_matrix();
        #[rustfmt::skip]
        let expected = DMatrix::from_data(VecStorage::new(Dyn(4), Dyn(4), vec![
            0.0, 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0, -2.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, -4.0,
        ])).transpose();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_roots() {
        let p = Polynom::<f64>::from_iter([-1.0, 0.0, 1.0]);
        let actual = p.roots();
        let expected = vec![-1.0, 1.0];
        assert_eq!(expected, actual);
    }
}
