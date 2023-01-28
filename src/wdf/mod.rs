use std::any::Any;
use std::cell::RefMut;
use std::{fmt, ops};

use std::{
    cell::RefCell,
    rc::{Rc},
};

use num_traits::Zero;
use numeric_literals::replace_float_literals;

// use crate::wdf::adaptors::Series;
use crate::Scalar;

mod adaptors;
mod leaves;
mod root;

#[derive(Debug, Copy, Clone)]
pub struct Wave<T> {
    pub a: T,
    pub b: T,
}

impl<T: Zero> Default for Wave<T> {
    fn default() -> Self {
        Self {
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> Wave<T> {
    pub fn from_kirchhoff(rp: T, v: T, i: T) -> Self {
        Self {
            a: v + rp * i,
            b: v - rp * i,
        }
    }
    pub fn a(a: T) -> Self {
        Self { a, b: T::zero() }
    }
    pub fn b(b: T) -> Self {
        Self { a: T::zero(), b }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn voltage(&self) -> T {
        0.5 * (self.a + self.b)
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn current(&self, rp: T) -> T {
        (self.a - self.b) * (0.5 / rp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Impedance<T> {
    r: T,
    g: T,
}

impl<T: Scalar> Impedance<T> {
    pub fn nonadaptable() -> Self {
        Self {
            r: T::zero(),
            g: T::infinity(),
        }
    }

    pub fn from_resistance(r: T) -> Self  {
        Self {
            r,
            g: r.recip(),
        }
    }

    pub fn from_admittance(g: T) -> Self {
        Self {
            r: g.recip(),
            g,
        }
    }

    pub fn set_resistance(&mut self, r: T) {
        self.r = r;
        self.g = r.recip();
    }

    pub fn set_admittance(&mut self, g: T) {
        self.g = g;
        self.r = g.recip();
    }

    pub fn is_adaptable(&self) -> bool {
        self.r.is_zero() || self.g.is_infinite()
    }

    #[inline]
    pub fn resistance(&self) -> T { self.r}

    #[inline]
    pub fn admittance(&self) -> T { self.g }

    pub fn parallel_with(self, other: Self) -> Self {
        Self {
            r: self.r*other.r / (self.r + other.r),
            g: self.g + other.g
        }
    }

    pub fn series_with(self, other: Self) -> Self {
        Self::from_resistance(self.r + other.r)
    }
}

#[allow(unused_variables)]
pub trait Wdf<T>: Any {
    fn reflected(&mut self, impedance: Impedance<T>, wave: &mut Wave<T>) -> T;
    fn incident(&mut self, impedance: Impedance<T>, wave: &mut Wave<T>, a: T);
    fn impedance(&self) -> Impedance<T>;
}

pub trait IntoNode<T>: Wdf<T> {
    fn into_node(self) -> Node<T, Self>
    where
        Self: Sized,
        T: Zero,
    {
        Node::from(self)
    }
}

impl<T, W: Wdf<T>> IntoNode<T> for W {}

#[repr(transparent)]
struct Invalidator(Box<dyn Fn()>);

impl<F: 'static + Fn()> From<F> for Invalidator {
    fn from(func: F) -> Self {
        Self(Box::new(func))
    }
}

impl fmt::Debug for Invalidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Invalidator").field(&"|| ...").finish()
    }
}

impl Invalidator {
    fn invalidate(&self) {
        (self.0)()
    }
}

/* impl<'a, T> Wdf<T> for &'a mut DynWdf<T> {
    fn impedance(&self) -> Impedance<T> {
        self.0.impedance()
    }

    fn reflected(&mut self, impedance: Impedance<T>, wave: &mut Wave<T>, a: T) -> T {
        self.0.reflected(impedance, wave, a)
    }

    fn incident(&mut self, impedence: Impedance<T>, wave: &mut Wave<T>, a: T) {
        self.0.incident(impedence, wave, a)
    }
} */

#[derive(Debug)]
struct NodeImpl<T, W: ?Sized> {
    pub invalidate_parent: Option<Invalidator>,
    pub wave: Wave<T>,
    pub impedance: Impedance<T>,
    pub wdf: W,
}

#[derive(Debug)]
pub struct Node<T, W: ?Sized>(Rc<RefCell<NodeImpl<T, W>>>);

impl<T, W: ?Sized> Clone for Node<T, W> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: 'static, W: ?Sized> Node<T, W> {
    #[inline]
    pub fn set_parent<W2: Wdf<T> + ?Sized>(&mut self, parent: &Node<T, W2>) {
        // This contrived snippet of code acts upon the invalidation by re-calculating the impedance value of the parent adaptor,
        // then bubbling the invalidation up the tree.
        let parent = Rc::downgrade(&parent.0);
        let mut b = self.0.borrow_mut();
        b.invalidate_parent.replace(Invalidator::from(move || {
            if let Some(parent) = parent.upgrade() {
                let mut pb = parent.borrow_mut();
                if let Some(invalidator) = &pb.invalidate_parent {
                    invalidator.invalidate();
                }
                pb.impedance = pb.wdf.impedance();
            }
        }));
    }

    #[inline]
    pub fn invalidate_parent_impedence(&self) {
        let b = self.0.borrow();
        if let Some(invalidator) = &b.invalidate_parent {
            invalidator.invalidate();
        }
    }

    pub fn inner(&self) -> NodeWdfRef<T, W> {
        NodeWdfRef { borrow: self.0.borrow_mut() }
    }
}

impl<'a, T, W> ops::DerefMut for NodeWdfRef<'a, T, W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.borrow.wdf
    }
}

impl<T: Copy, W: ?Sized> Node<T, W> {
    #[inline]
    pub fn wave(&self) -> Wave<T> {
        self.0.borrow().wave
    }

    #[inline]
    pub fn impedance(&self) -> Impedance<T> {
        self.0.borrow().impedance
    }
}

impl<T: 'static + Copy, W: Wdf<T> + ?Sized> Node<T, W> {
    #[inline]
    pub fn reflected(&mut self) -> T {
        let mut b = self.0.borrow_mut();
        let b = &mut *b;
        let wave = &mut b.wave;
        b.wdf.reflected(b.impedance, wave)
    }

    #[inline]
    pub fn incident(&mut self, a: T) {
        let mut b = self.0.borrow_mut();
        let b = &mut *b;
        let wave = &mut b.wave;
        b.wdf.incident(b.impedance, wave, a)
    }
}

impl<T: Zero, W: Wdf<T>> From<W> for Node<T, W> {
    fn from(wdf: W) -> Self {
        let impedance = wdf.impedance();
        Self(Rc::new(RefCell::new(NodeImpl {
            invalidate_parent: None,
            wave: Wave::default(),
            impedance,
            wdf,
        })))
    }
}

impl<T: Scalar, W: Wdf<T>> Node<T, W> {
    #[replace_float_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn voltage(&self) -> T {
        self.0.borrow().wave.voltage()
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn current(&self) -> T {
        let b = self.0.borrow();
        let rp = b.wdf.impedance().resistance();
        b.wave.current(rp)
    }
}

pub struct NodeWdfRef<'a, T, W: ?Sized> {
    borrow: RefMut<'a, NodeImpl<T, W>>
}

impl<'a, T, W> ops::Deref for NodeWdfRef<'a, T, W> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.borrow.wdf
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicBool};

    use num_traits::Zero;

    use crate::{wdf::{
        adaptors::{Series},
        leaves::{Resistor},
        root::IdealVs,
        IntoNode, Wdf, Node,
    }};

    use super::{Impedance, adaptors::PolarityInvert};

    #[test]
    /// Tests that the invalidation behavior bubbles up the WDF tree
    fn bubbling_invalidation() {
        struct MonitorImpedance<T, W> {
            impedance_called: AtomicBool,
            child_node: Node<T, W>
        }

        impl<T: Zero, W> MonitorImpedance<T, W> where Self: Wdf<T> {
            fn new(child: impl Into<Node<T, W>>) -> Node<T, Self> {
                let mut child = child.into();
                let this = Self { child_node: child.clone(), impedance_called: AtomicBool::new(false) }.into_node();
                child.set_parent(&this);
                this
            }

            fn impedance_called(&self) -> bool {
                self.impedance_called.load(std::sync::atomic::Ordering::SeqCst)
            }
        }

        impl<T:'static + Copy, W: Wdf<T>> Wdf<T> for MonitorImpedance<T, W> {
            fn impedance(&self) -> Impedance<T> {
                self.impedance_called.store(true, std::sync::atomic::Ordering::SeqCst);
                self.child_node.impedance()
            }

            fn reflected(&mut self, _impedance: Impedance<T>, _wave: &mut crate::wdf::Wave<T>) -> T {
                self.child_node.reflected()
            }

            fn incident(&mut self, _impedance: Impedance<T>, _wave: &mut crate::wdf::Wave<T>, a: T) {
                self.child_node.incident(a)
            }
        }

        let child = Resistor(Impedance::from_resistance(330f32)).into_node();
        let monitor = MonitorImpedance::new(child.clone());
        child.invalidate_parent_impedence();
        assert!(monitor.inner().impedance_called());
    }

    #[test]
    fn voltage_divider() {
        let r1 = Resistor(Impedance::from_resistance(10e3)).into_node();
        let r2 = Resistor(Impedance::from_resistance(10e3)).into_node();
        let s1 = Series::new(r1, r2.clone());
        let mut p1 = PolarityInvert::new(s1);
        let mut vs = IdealVs::new(p1.clone());
        vs.inner().vs = 1e3;
        vs.incident(p1.reflected());
        p1.incident(vs.reflected());
        let vout = r2.voltage();
        assert_eq!(vout, 500.);
    }
}
