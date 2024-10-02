fn main() {
    use abrasive::Abrasive;
    use nih_plug::wrapper::standalone::nih_export_standalone;

    nih_export_standalone::<Abrasive>();
}
