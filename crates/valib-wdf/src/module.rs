use crate::dsl::node_mut;
use crate::{AdaptedWdf, Node, Wdf};

pub struct WdfModule<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> {
    pub root: Node<Root>,
    pub leaf: Node<Leaf>,
}

impl<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> WdfModule<Root, Leaf> {
    pub fn new(root: Node<Root>, leaf: Node<Leaf>) -> Self {
        Self { root, leaf }
    }

    pub fn set_samplerate(&mut self, samplerate: f64) {
        let mut root = node_mut(&self.root);
        let mut leaf = node_mut(&self.leaf);
        root.set_samplerate(samplerate);
        leaf.set_samplerate(samplerate);
    }

    pub fn process_sample(&mut self) {
        let mut root = node_mut(&self.root);
        let mut leaf = node_mut(&self.leaf);
        root.set_port_resistance(leaf.impedance());
        root.incident(leaf.reflected());
        leaf.incident(root.reflected());
    }

    pub fn reset(&mut self) {
        node_mut(&self.root).reset();
        node_mut(&self.leaf).reset();
    }
}
