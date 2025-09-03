import asyncio
import time
from asyncio import sleep
from prompt_toolkit.application import get_app

from .state import State
from .fetch import audiofetch
from .layout import app

def update_fps():
    now = time.time()
    delta = now - State._last_time
    State._last_time = now
    if delta > 0:
        State.fps = round(1 / delta, 1)
    State.frame = (State.frame + 1) % 10000000

async def render_loop():
    while True:
        update_fps()
        get_app().invalidate()
        await sleep(1 / State.target_fps)

async def main():
    audiofetch()
    render_task = asyncio.create_task(render_loop())
    try:
        await app.run_async()
    finally:
        render_task.cancel()

def entrypoint():
    asyncio.run(main())