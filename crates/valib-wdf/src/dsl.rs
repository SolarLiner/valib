//! # WDF DSL utilities
//!
//! Utility module exposing the WDF node constructors as freestanding functions, allowing one to
//! quickly compose a WDF tree together.
use crate::*;
use atomic_refcell::AtomicRefCell;
use std::sync::Arc;
use valib_core::Scalar;
use valib_saturators::clippers::{DiodeClipper, DiodeClipperModel};

/// Create a new node by wrapping the raw node data into a shared node.
#[inline]
pub fn node<T>(value: T) -> Node<T> {
    Arc::new(AtomicRefCell::new(value))
}

/// Return a reference to the node data by borrowing the contents.
#[inline]
pub fn node_ref<T>(value: &Node<T>) -> NodeRef<T> {
    value.borrow()
}

/// Return a mutable reference to the node data by borrowing the contents.
#[inline]
pub fn node_mut<T>(value: &Node<T>) -> NodeMut<T> {
    value.borrow_mut()
}

/// Compute the voltage at the upper facing port of the provided node.
#[inline]
pub fn voltage<T: Scalar>(node: &Node<impl Wdf<Scalar = T>>) -> T {
    node_ref(node).wave().voltage()
}

/// Compute the current at the upper facing port of the provided node.
#[inline]
pub fn current<T: Scalar>(node: &Node<impl AdaptedWdf<Scalar = T>>) -> T {
    let n = node_ref(node);
    n.wave().current(n.impedance())
}

/// Create a new resistor.
///
/// See [`Resistor::new`] for more details.
#[inline]
pub fn resistor<T: Scalar>(r: T) -> Node<Resistor<T>> {
    node(Resistor::new(r))
}

/// Create a new capacitor.
///
/// See [`Capacitor::new`] for more details.
#[inline]
pub fn capacitor<T: Scalar>(fs: T, c: T) -> Node<Capacitor<T>> {
    node(Capacitor::new(fs, c))
}

/// Create a new resistive voltage source.
///
/// See [`ResistiveVoltageSource::new`] for more details.
#[inline]
pub fn rvsource<T: Scalar>(r: T, vs: T) -> Node<ResistiveVoltageSource<T>> {
    node(ResistiveVoltageSource::new(r, vs))
}

/// Create a new ideal voltage source.
///
/// See [`IdealVoltageSource::new`] for more details.
#[inline]
pub fn ivsource<T: Zero>(vs: T) -> Node<IdealVoltageSource<T>> {
    node(IdealVoltageSource::new(vs))
}

/// Create a new resistive current source.
///
/// See [`ResistiveCurrentSource::new`] for more details.
#[inline]
pub fn rcsource<T: Zero>(r: T, j: T) -> Node<ResistiveCurrentSource<T>> {
    node(ResistiveCurrentSource::new(r, j))
}

/// Create a new ideal current source.
///
/// See [`IdealCurrentSource::new`] for more details.
#[inline]
pub fn icsource<T: Zero>(j: T) -> Node<IdealCurrentSource<T>> {
    node(IdealCurrentSource::new(j))
}

/// Create a new short circuit node.
///
/// See [`ShortCircuit::new`] for more details.
#[inline]
pub fn short_circuit<T: Zero>() -> Node<ShortCircuit<T>> {
    node(ShortCircuit::default())
}

/// Create a new open circuit node.
///
/// See [`OpenCircuit::new`] for more details.
#[inline]
pub fn open_circuit<T: Zero>() -> Node<OpenCircuit<T>> {
    node(OpenCircuit::default())
}

/// Create a new dsp wdf node.
///
/// See [`WdfDsp::new`] for more details.
#[inline]
pub fn dsp<P: DSPMeta<Sample: Zero>>(dsp: P) -> Node<WdfDsp<P>> {
    node(WdfDsp::new(dsp))
}

/// Create a new series wdf adapter node.
///
/// See [`Series::new`] for more details.
#[inline]
pub fn series<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>>(
    left: Node<A>,
    right: Node<B>,
) -> Node<Series<A, B>> {
    node(Series::new(left, right))
}

/// Create a new parallel wdf adapter node.
///
/// See [`Parallel::new`] for more details.
#[inline]
pub fn parallel<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>>(
    left: Node<A>,
    right: Node<B>,
) -> Node<Parallel<A, B>> {
    node(Parallel::new(left, right))
}

/// Create a new polarity inverter wdf adapter node.
///
/// See [`Inverter::new`] for more details.
#[inline]
pub fn inverter<W: AdaptedWdf>(inner: Node<W>) -> Node<Inverter<W>> {
    node(Inverter::new(inner))
}

/// Create a new Lambert W function-based diode clipper node.
///
/// See [`DiodeLambert::new`] for more details.
#[inline]
pub fn diode_lambert<T: Scalar>(data: DiodeClipper<T>) -> Node<DiodeLambert<T>> {
    node(DiodeLambert::new(data))
}

/// Create a new analytical model-based diode clipper node.
#[inline]
pub fn diode_model<T: Scalar>(model: DiodeClipperModel<T>) -> Node<DiodeModel<T>> {
    dsp(model)
}

/// Create a new Newton-Rhapson-based diode clipper node.
///
/// See [`DiodeNR::new`] for more details.
#[inline]
pub fn diode_nr<T: Scalar>(data: DiodeClipper<T>) -> Node<DiodeNR<T>> {
    node(DiodeNR::from_data(data))
}

/// Create a new WDF module with the provided node and leaf types.
///
/// See [`WdfModule::new`] for more details.
#[inline]
pub fn module<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>>(
    root: Node<Root>,
    leaf: Node<Leaf>,
) -> WdfModule<Root, Leaf> {
    WdfModule::new(root, leaf)
}
