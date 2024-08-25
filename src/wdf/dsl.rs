use crate::wdf::{AdaptedWdf, Node, NodeMut, NodeRef, Wdf};
use crate::Scalar;
use atomic_refcell::AtomicRefCell;
use std::sync::Arc;

#[inline]
pub fn node<T>(value: T) -> Node<T> {
    Arc::new(AtomicRefCell::new(value))
}

#[inline]
pub fn node_ref<T>(value: &Node<T>) -> NodeRef<T> {
    value.borrow()
}

#[inline]
pub fn node_mut<T>(value: &Node<T>) -> NodeMut<T> {
    value.borrow_mut()
}

#[inline]
pub fn voltage<T: Scalar>(node: &Node<impl Wdf<Scalar = T>>) -> T {
    node_ref(node).wave().voltage()
}

#[inline]
pub fn current<T: Scalar>(node: &Node<impl AdaptedWdf<Scalar = T>>) -> T {
    let n = node_ref(node);
    n.wave().current(n.impedance())
}
