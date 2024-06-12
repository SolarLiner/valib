use std::sync::atomic::Ordering;
use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_iced::{
    create_iced_editor, executor, widgets, Alignment, Column, Command, Element, IcedEditor,
    IcedState, Length, Text, WindowQueue,
};

use widgets::generic_ui::{GenericSlider, GenericUi};

use crate::editor::led::Led;
use crate::params::Ts404Params;

mod led;

pub(crate) fn default_state() -> Arc<IcedState> {
    IcedState::from_size(200, 300)
}

pub(crate) fn create(
    params: Arc<Ts404Params>,
    drive_led: Arc<AtomicF32>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<Ts404Editor>(params.editor_state.clone(), (params, drive_led))
}

struct Ts404Editor {
    params: Arc<Ts404Params>,
    context: Arc<dyn GuiContext>,
    generic_ui: widgets::generic_ui::State<GenericSlider>,
    drive_led: Arc<AtomicF32>,
}

#[derive(Debug, Copy, Clone)]
enum EditorMessage {
    ParamUpdate(widgets::ParamMessage),
}

impl IcedEditor for Ts404Editor {
    type Executor = executor::Default;
    type Message = EditorMessage;
    type InitializationFlags = (Arc<Ts404Params>, Arc<AtomicF32>);

    fn new(
        (params, drive_led): Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let editor = Self {
            params,
            context,
            drive_led,
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
            .push(
                GenericUi::new(&mut self.generic_ui, self.params.clone())
                    .width(Length::Units(200))
                    .height(Length::Fill)
                    .map(EditorMessage::ParamUpdate),
            )
            .push(Led::new(led / 20.0))
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}
