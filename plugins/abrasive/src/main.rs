use nih_plug::wrapper::standalone::nih_export_standalone;
use abrasive::Abrasive;

fn main() {
    nih_export_standalone::<Abrasive<2, {abrasive::NUM_BANDS}>>();
}