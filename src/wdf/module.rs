use crate::wdf::{AdaptedWdf, Node, Wdf};

pub struct WdfModule<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> {
    pub root: Node<Root>,
    pub leaf: Node<Leaf>,
}

impl<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> WdfModule<Root, Leaf> {
    pub fn new(root: Node<Root>, leaf: Node<Leaf>) -> Self {
        Self { root, leaf }
    }

    pub fn set_samplerate(&mut self, samplerate: f64) {
        let mut root = self.root.borrow_mut();
        let mut leaf = self.leaf.borrow_mut();
        root.set_samplerate(samplerate);
        leaf.set_samplerate(samplerate);
    }

    pub fn next_sample(&mut self) {
        let mut root = self.root.borrow_mut();
        let mut leaf = self.leaf.borrow_mut();
        root.set_port_resistance(leaf.impedance());
        root.incident(leaf.reflected());
        leaf.incident(root.reflected());
    }

    pub fn reset(&mut self) {
        self.root.borrow_mut().reset();
        self.leaf.borrow_mut().reset();
    }
}
