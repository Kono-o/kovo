from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.output.color_depth import ColorDepth

from .fmt import wave_fmt

def box(renderer, height=None, width=None):
    b = None
    def render():
        return renderer(b)
    b = Window(
        content=FormattedTextControl(render),
        height=height,
        width=width,
    )
    return b

wave_box = box(wave_fmt)

layout = Layout(VSplit([wave_box]))
layout.focus_stack = []

kb = KeyBindings()
@kb.add("q")
def exit_(event):
    event.app.exit()

app = Application(
    layout=layout,
    key_bindings=kb,
    color_depth=ColorDepth.TRUE_COLOR,
    full_screen=True,
)
