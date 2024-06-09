use std::sync::Arc;
use std::sync::atomic::Ordering;

use nih_plug::prelude::*;
use nih_plug_iced::{
    Alignment, Column, Command, create_iced_editor, Element, executor, IcedEditor, IcedState,
    Length, Text, widgets, WindowQueue,
};
use nih_plug_iced::futures::channel::mpsc::Receiver;
use widgets::generic_ui::{GenericSlider, GenericUi};

use crate::editor::led::Led;
use crate::params::Ts404Params;

mod led;

pub(crate) fn default_state() -> Arc<IcedState> {
    IcedState::from_size(120, 300)
}

pub(crate) fn create(
    params: Arc<Ts404Params>,
    driver_led_receiver: Receiver<Arc<AtomicF32>>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<Ts404Editor>(params.editor_state.clone(), (params, driver_led_receiver))
}

struct Ts404Editor {
    params: Arc<Ts404Params>,
    context: Arc<dyn GuiContext>,
    generic_ui: widgets::generic_ui::State<GenericSlider>,
    drive_led: Arc<AtomicF32>,
    driver_led_receiver: Receiver<Arc<AtomicF32>>,
}

#[derive(Debug, Copy, Clone)]
enum EditorMessage {
    ParamUpdate(widgets::ParamMessage),
}

impl IcedEditor for Ts404Editor {
    type Executor = executor::Default;
    type Message = EditorMessage;
    type InitializationFlags = (Arc<Ts404Params>, Receiver<Arc<AtomicF32>>);

    fn new(
        (params, driver_led_receiver): Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let editor = Self {
            params,
            context,
            driver_led_receiver,
            generic_ui: Default::default(),
        };
        (editor, Command::none())
    }

    fn context(&self) -> &dyn GuiContext {
        self.context.as_ref()
    }

    fn update(
        &mut self,
        _window: &mut WindowQueue,
        message: Self::Message,
    ) -> Command<Self::Message> {
        match message {
            EditorMessage::ParamUpdate(msg) => self.handle_param_message(msg),
        }
        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        let led = self.drive_led.load(Ordering::SeqCst);
        nih_log!("Editor view: {led}");
        Column::new()
            .align_items(Alignment::Fill)
            .push(
                GenericUi::new(&mut self.generic_ui, self.params.clone())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .map(EditorMessage::ParamUpdate),
            )
            .push(Text::new(format!("LED value: {led}")))
            .push(Led::new(led))
            .into()
    }
}
