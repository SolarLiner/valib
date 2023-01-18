use numeric_literals::replace_float_literals;
use crate::Scalar;

#[replace_float_literals(T::from(literal).unwrap())]
pub fn lerp_block<T: Scalar>(out: &mut [T], inp: &[T]) {
    let rate = T::from(inp.len()).unwrap() / T::from(out.len()).unwrap();

    for (i, y) in out.iter_mut().enumerate() {
        let j = T::from(i).unwrap() * rate;
        let f = j.fract();
        let j = j.floor().to_usize().unwrap();
        let a = inp[j];
        let b = inp.get(j + 1).copied().unwrap_or(a);
        *y = lerp(f, a, b);
    }
}

pub fn lerp<T: Scalar>(t: T, a: T, b: T) -> T {
    a + t * (b - a)
}

#[cfg(test)]
mod tests {
    use super::lerp_block;

    #[test]
    fn interp_block() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1.];
        lerp_block(&mut actual, &a);
        assert_eq!(actual, expected);
    }
}
