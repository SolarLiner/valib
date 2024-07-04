use nih_plug::nih_export_standalone;
use ts404::Ts404;

fn main() {
    profiling::register_thread!("Main thread");
    nih_export_standalone::<Ts404>();
}