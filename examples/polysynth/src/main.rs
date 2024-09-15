use nih_plug::nih_export_standalone;
use polysynth::PolysynthPlugin;

fn main() {
    nih_export_standalone::<PolysynthPlugin>();
}
