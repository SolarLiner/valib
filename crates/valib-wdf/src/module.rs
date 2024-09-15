//! # WDF module
//!
//! Provides a module which can drive the entire WDF tree for each sample.
use crate::dsl::node_mut;
use crate::{AdaptedWdf, Node, Wdf};

/// WDF Module type. This type takes care of processing the whole tree when processing a sample.
///
/// It does not take care of inputs and outputs; they should be manually set and manually read by
/// cloning relevant nodes and reading/mutating them.
pub struct WdfModule<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> {
    /// Root of the tree. Can be unadaptable.
    pub root: Node<Root>,
    /// Leaf of the tree. Has to be adaptable.
    pub leaf: Node<Leaf>,
}

impl<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> WdfModule<Root, Leaf> {
    /// Create a new WDF module.
    ///
    /// # Arguments
    ///
    /// * `root`: Root of the tree. Can be unadaptable.
    /// * `leaf`: Leaf of the tree. Has to be adaptable.
    ///
    /// returns: WdfModule<Root, Leaf>
    pub fn new(root: Node<Root>, leaf: Node<Leaf>) -> Self {
        Self { root, leaf }
    }

    /// Sets the sample rate of the whole module. Calls [`Wdf::set_samplerate`] on both the root and
    /// the leaf.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: New sample rate (Hz)
    ///
    /// returns: ()
    pub fn set_samplerate(&mut self, samplerate: f64) {
        let mut root = node_mut(&self.root);
        let mut leaf = node_mut(&self.leaf);
        root.set_samplerate(samplerate);
        leaf.set_samplerate(samplerate);
    }

    /// Process a single sample, propagating all waves downwards and back up.
    pub fn process_sample(&mut self) {
        let mut root = node_mut(&self.root);
        let mut leaf = node_mut(&self.leaf);
        root.set_port_resistance(leaf.impedance());
        root.incident(leaf.reflected());
        leaf.incident(root.reflected());
    }

    /// Reset the entire module. Calls [`Wdf::reset`] on both the root and the leaf.
    pub fn reset(&mut self) {
        node_mut(&self.root).reset();
        node_mut(&self.leaf).reset();
    }
}
