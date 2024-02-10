#[cfg(not(feature = "example"))]
fn main() {
    use abrasive::Abrasive;
    use nih_plug::wrapper::standalone::nih_export_standalone;

    nih_export_standalone::<Abrasive<{ abrasive::NUM_BANDS }>>();
}

#[cfg(feature = "example")]
fn main() {
    panic!("Use the example binary instead of the main one");
}
