use crate::Scalar;
use num_traits::Zero;
use petgraph::prelude::*;
use petgraph::visit::GraphBase;
use std::any::{type_name, Any};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::{fmt, hash, ops};

#[derive(Debug, Copy, Clone)]
pub struct Port<T> {
    pub a: T,
    pub b: T,
    pub rp: T,
}

impl<T: Zero> Default for Port<T> {
    fn default() -> Self {
        Self {
            a: T::zero(),
            b: T::zero(),
            rp: T::zero(),
        }
    }
}

impl<T: Scalar> Port<T> {
    pub fn voltage(&self) -> T {
        (self.a + self.b) / T::from_f64(2.0)
    }

    pub fn current(&self) -> T {
        (self.a - self.b) / (T::from_f64(2.0) * self.rp)
    }
}

pub trait Wdf: Any {
    type Scalar: Scalar;

    fn num_ports(&self) -> usize;

    fn scatter(&mut self, ports: &mut dyn ops::IndexMut<usize, Output = Port<Self::Scalar>>);
}

pub trait OnePort: Any {
    type Scalar: Scalar;

    fn reflect(&mut self, port: &mut Port<Self::Scalar>);
}

impl<P: OnePort> Wdf for P {
    type Scalar = P::Scalar;

    fn num_ports(&self) -> usize {
        1
    }

    fn scatter(&mut self, ports: &mut dyn ops::IndexMut<usize, Output = Port<Self::Scalar>>) {
        P::reflect(self, &mut ports[0]);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Resistor<T>(pub T);

impl<T: Scalar> OnePort for Resistor<T> {
    type Scalar = T;

    fn reflect(&mut self, port: &mut Port<Self::Scalar>) {
        port.b = T::zero();
        port.rp = self.0;
    }
}

#[derive(Debug, Copy, Clone)]
pub struct IdealVoltageSource<T>(pub T);

impl<T: Scalar> OnePort for IdealVoltageSource<T> {
    type Scalar = T;

    fn reflect(&mut self, port: &mut Port<Self::Scalar>) {
        port.b = -port.a + T::from_f64(2.0) * self.0;
    }
}

#[derive(Debug, Copy, Clone)]
struct GraphEdge<T> {
    node_src: usize,
    port_src: usize,
    node_dst: usize,
    port_dst: usize,
    port: Port<T>,
}

impl<T> GraphEdge<T> {
    fn has_port(&self, node_ix: usize, port_ix: usize) -> bool {
        self.node_src == node_ix && self.port_src == port_ix
            || self.node_dst == node_ix && self.port_dst == port_ix
    }

    fn get_port(&self, node_ix: usize, port_ix: usize) -> Option<Port<T>>
    where
        T: Copy,
    {
        self.has_port(node_ix, port_ix).then_some(self.port)
    }
}

type Graph<T> = DiGraph<usize, GraphEdge<T>>;
type NodeId<T> = <Graph<T> as GraphBase>::NodeId;

pub struct Index<T: Wdf> {
    index: usize,
    node_id: NodeId<T>,
}

impl<T: Wdf> fmt::Debug for Index<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct(type_name::<Self>())
            .field("index", &self.index)
            .field("node_id", &self.node_id)
            .finish()
    }
}

impl<T: Wdf> Clone for Index<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            node_id: self.node_id.clone(),
        }
    }
}

impl<T: Wdf> Copy for Index<T> {}

pub struct WdfTree<T: Scalar> {
    data: Vec<Box<dyn Wdf<Scalar = T>>>,
    graph: Graph<T>,
    sort: Vec<NodeId<T>>,
}

impl<T: Scalar, W: Wdf<Scalar = T>> ops::Index<Index<W>> for WdfTree<T> {
    type Output = W;

    fn index(&self, index: Index<W>) -> &Self::Output {
        self.get(index).expect(&format!(
            "Cannot get a {} from index {}",
            type_name::<W>(),
            index.index
        ))
    }
}

impl<T: Scalar> WdfTree<T> {
    pub fn new() -> Self {
        Self {
            data: vec![],
            graph: Graph::<T>::default(),
            sort: vec![],
        }
    }

    pub fn add_node<W: Wdf<Scalar = T>>(&mut self, node: W) -> Index<W> {
        let index = self.data.len();
        self.data.push(Box::new(node));
        let node_id = self.graph.add_node(index);
        Index { index, node_id }
    }

    pub fn add_connection<W1: Wdf<Scalar = T>, W2: Wdf<Scalar = T>>(
        &mut self,
        left: Index<W1>,
        port_left: usize,
        right: Index<W2>,
        port_right: usize,
    ) {
        let e = GraphEdge {
            node_src: left.index,
            port_src: port_left,
            node_dst: right.index,
            port_dst: port_right,
            port: Port::default(),
        };
        self.graph.add_edge(left.node_id, right.node_id, e);
    }

    pub fn get<W: Wdf<Scalar = T>>(&self, index: Index<W>) -> Option<&W> {
        if cfg!(debug_assertions) {
            let any: &dyn Any = &self.data[index.index];
            match any.downcast_ref::<W>() {
                None => None,
                Some(value) => Some(value),
            }
        } else {
            unsafe {
                let any: &dyn Any = &*self.data.get_unchecked(index.index);
                Some(any.downcast_ref_unchecked::<W>())
            }
        }
    }

    pub fn get_mut<W: Wdf<Scalar = T>>(&mut self, index: Index<W>) -> Option<&mut W> {
        if cfg!(debug_assertions) {
            let any: &mut dyn Any = &mut self.data[index.index];
            match any.downcast_mut::<W>() {
                None => {
                    panic!(
                        "Cannot index for {} at index {}",
                        type_name::<W>(),
                        index.index
                    )
                }
                Some(value) => Some(value),
            }
        } else {
            unsafe {
                let any: &mut dyn Any = &mut *self.data.get_unchecked_mut(index.index);
                Some(any.downcast_mut_unchecked::<W>())
            }
        }
    }

    pub fn get_port<W: Wdf<Scalar = T>>(&self, index: Index<W>, port: usize) -> Option<Port<T>> {
        self.graph
            .edges(index.node_id)
            .find_map(|edge| edge.weight().get_port(index.index, port))
    }

    pub fn process_sample(&mut self) {
        struct GetPort<'a, T> {
            graph: &'a mut Graph<T>,
            node_id: NodeId<T>,
        }

        impl<'a, T> ops::Index<usize> for GetPort<'a, T> {
            type Output = Port<T>;

            fn index(&self, index: usize) -> &Self::Output {
                let node_ix = self.graph[self.node_id];
                let edge = self
                    .graph
                    .edges(self.node_id)
                    .find_map(|edge| edge.weight().has_port(node_ix, index).then_some(edge.id()))
                    .unwrap();
                &self.graph[edge].port
            }
        }

        impl<'a, T> ops::IndexMut<usize> for GetPort<'a, T> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                let node_ix = self.graph[self.node_id];
                let edge = self
                    .graph
                    .edges(self.node_id)
                    .find_map(|edge| edge.weight().has_port(node_ix, index).then_some(edge.id()))
                    .unwrap();
                &mut self.graph[edge].port
            }
        }

        if self.sort.len() != self.graph.node_count() {
            self.sort = petgraph::algo::toposort(&self.graph, None).unwrap();
        }
        for node_id in self.sort.iter().copied() {
            let node = self.graph.node_weight(node_id).copied().unwrap();
            let data = &mut *self.data[node];
            let mut get_port = GetPort {
                graph: &mut self.graph,
                node_id,
            };
            data.scatter(&mut get_port);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voltage_divider() {
        let mut tree = WdfTree::new();
        let src = tree.add_node(IdealVoltageSource(10.0));
        let res = tree.add_node(Resistor(100.0));
        tree.add_connection(src, 0, res, 0);
        tree.process_sample();
        let port = tree.get_port(res, 0).unwrap();
        assert_eq!(10.0, port.voltage());
        assert_eq!(100.0e-3, port.current());
    }
}
