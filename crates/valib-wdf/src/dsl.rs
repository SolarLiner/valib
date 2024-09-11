use crate::saturators::clippers::{DiodeClipper, DiodeClipperModel};
use crate::wdf::*;
use atomic_refcell::AtomicRefCell;
use std::sync::Arc;
use valib_core::Scalar;

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

#[inline]
pub fn resistor<T: Scalar>(r: T) -> Node<Resistor<T>> {
    node(Resistor::new(r))
}

#[inline]
pub fn capacitor<T: Scalar>(fs: T, c: T) -> Node<Capacitor<T>> {
    node(Capacitor::new(fs, c))
}

#[inline]
pub fn rvsource<T: Scalar>(r: T, vs: T) -> Node<ResistiveVoltageSource<T>> {
    node(ResistiveVoltageSource::new(r, vs))
}

#[inline]
pub fn ivsource<T: Zero>(vs: T) -> Node<IdealVoltageSource<T>> {
    node(IdealVoltageSource::new(vs))
}

#[inline]
pub fn rcsource<T: Zero>(r: T, j: T) -> Node<ResistiveCurrentSource<T>> {
    node(ResistiveCurrentSource::new(j, r))
}

#[inline]
pub fn icsource<T: Zero>(j: T) -> Node<IdealCurrentSource<T>> {
    node(IdealCurrentSource::new(j))
}

#[inline]
pub fn short_circuit<T: Zero>() -> Node<ShortCircuit<T>> {
    node(ShortCircuit::default())
}

#[inline]
pub fn open_circuit<T: Zero>() -> Node<OpenCircuit<T>> {
    node(OpenCircuit::default())
}

#[inline]
pub fn dsp<P: DSPMeta<Sample: Zero>>(dsp: P) -> Node<WdfDsp<P>> {
    node(WdfDsp::new(dsp))
}

#[inline]
pub fn series<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>>(
    left: Node<A>,
    right: Node<B>,
) -> Node<Series<A, B>> {
    node(Series::new(left, right))
}

#[inline]
pub fn parallel<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>>(
    left: Node<A>,
    right: Node<B>,
) -> Node<Parallel<A, B>> {
    node(Parallel::new(left, right))
}

#[inline]
pub fn inverter<W: AdaptedWdf>(inner: Node<W>) -> Node<Inverter<W>> {
    node(Inverter::new(inner))
}

#[inline]
pub fn diode_lambert<T: Scalar>(data: DiodeClipper<T>) -> Node<DiodeLambert<T>> {
    node(DiodeLambert::new(data))
}

#[inline]
pub fn diode_model<T: Scalar>(model: DiodeClipperModel<T>) -> Node<DiodeModel<T>> {
    dsp(model)
}

#[inline]
pub fn diode_nr<T: Scalar>(data: DiodeClipper<T>) -> Node<DiodeNR<T>> {
    node(DiodeNR::from_data(data))
}

#[inline]
pub fn module<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>>(
    root: Node<Root>,
    leaf: Node<Leaf>,
) -> WdfModule<Root, Leaf> {
    WdfModule::new(root, leaf)
}
