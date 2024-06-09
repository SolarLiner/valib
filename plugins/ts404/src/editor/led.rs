use nih_plug_iced::{Color, Element, Layout, Length, Point, Rectangle, renderer, Size, Widget};
use nih_plug_iced::layout::{Limits, Node};
use nih_plug_iced::renderer::{Quad, Style};

pub(crate) struct Led {
    size: Length,
    brightness: f32,
    color: Color,
}

impl Led {
    pub(crate) fn new(vin: f32) -> Self {
        Self {
            size: Length::Units(20),
            brightness: 1.0 - f32::exp(-vin),
            color: Color::from_rgb(1.0, 0.0, 0.0),
        }
    }

    pub(crate) fn size(mut self, size: Length) -> Self {
        self.size = size;
        self
    }

    pub(crate) fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

impl<Message, Renderer: renderer::Renderer> Widget<Message, Renderer> for Led {
    fn width(&self) -> Length {
        self.size
    }

    fn height(&self) -> Length {
        self.size
    }

    fn layout(&self, _renderer: &Renderer, limits: &Limits) -> Node {
        Node::new(
            limits
                .width(self.size)
                .height(self.size)
                .resolve(Size::ZERO),
        )
    }

    fn draw(
        &self,
        renderer: &mut Renderer,
        _style: &Style,
        layout: Layout<'_>,
        _cursor_position: Point,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let radius = bounds.width;
        let color = scale_color(self.color, self.brightness);
        renderer.fill_quad(Quad {
            bounds,
            border_radius: radius,
            border_width: 0.0,
            border_color: Color::TRANSPARENT,
        }, color);
    }
}

impl<'a, Message: 'a + Clone> From<Led> for Element<'a, Message> {
    fn from(value: Led) -> Self {
        Element::new(value)
    }
}

fn scale_color(col: Color, amt: f32) -> Color {
    let [r,g,b] = [col.r, col.g, col.b].map(|x| x * amt);
    Color::from_rgba(r, g, b, col.a)
}
