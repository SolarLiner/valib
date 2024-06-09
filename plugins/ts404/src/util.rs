use std::collections::VecDeque;
use nih_plug::nih_log;
use num_traits::Zero;
use valib::Scalar;

#[derive(Debug, Clone)]
pub struct Rms<T> {
    data: VecDeque<T>,
    summed_squared: T,
}

impl<T: Zero> Rms<T> {
    pub fn new(size: usize) -> Self {
        Self {
            data: (0..size).map(|_| T::zero()).collect(),
            summed_squared: T::zero(),
        }
    }
}

impl<T: Scalar> Rms<T> {
    pub fn add_element(&mut self, value: T) -> T {
        let v2 = value.simd_powi(2);
        self.summed_squared -= self.data.pop_front().unwrap();
        self.summed_squared += v2;
        self.data.push_back(v2);
        self.get_rms()
    }
    
    pub fn get_rms(&self) -> T {
        self.summed_squared.simd_sqrt()
    }
}