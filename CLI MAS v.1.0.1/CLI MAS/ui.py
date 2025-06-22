import shutil, textwrap, time
import colorama, threading
import random

colorama.init(strip=False, convert=True)

__all__ = ["print_jrpg_box", "loading_animation"]

# Expanded palette (skip WHITE to keep contrast)
COLOUR_PALETTE = [
    colorama.Fore.RED,
    colorama.Fore.GREEN,
    colorama.Fore.YELLOW,
    colorama.Fore.BLUE,
    colorama.Fore.MAGENTA,
    colorama.Fore.CYAN,
    colorama.Fore.LIGHTRED_EX,
    colorama.Fore.LIGHTGREEN_EX,
    colorama.Fore.LIGHTYELLOW_EX,
    colorama.Fore.LIGHTBLUE_EX,
    colorama.Fore.LIGHTMAGENTA_EX,
    colorama.Fore.LIGHTCYAN_EX,
]


def assign_colour(name: str, taken: dict[str, str]) -> str:
    """Return a persistent colour for *name*, reusing previous mapping or picking a
    random unused hue from COLOUR_PALETTE. Falls back to white if we run out."""

    if name in taken:
        return taken[name]

    available = [c for c in COLOUR_PALETTE if c not in taken.values()]
    colour = random.choice(available) if available else colorama.Fore.WHITE
    taken[name] = colour
    return colour


def loading_animation(stop_event: threading.Event, speaker_name: str, colour: str = "") -> None:
    """Simple three-dot spinner."""
    reset = colorama.Style.RESET_ALL
    frames = ["   ", ".  ", ".. ", "..."]
    i = 0
    while not stop_event.is_set():
        dots = frames[i % len(frames)]
        print(f"\r{colour}{speaker_name} is thinking{dots}{reset}", end="", flush=True)
        i += 1
        time.sleep(0.3)
    # Clear the entire line properly
    print("\r" + " " * (len(speaker_name) + 20) + "\r", end="", flush=True)


def print_jrpg_box(speaker_name: str, colour: str, text: str, tts_engine=None, voice_id=None, *, typewriter: bool = False) -> None:
    reset = colorama.Style.RESET_ALL
    try:
        term_w = shutil.get_terminal_size((80, 24)).columns
    except OSError:
        term_w = 80

    box_w = min(80, term_w)
    text_w = box_w - 4

    prefix = f"{speaker_name}: "
    lines = textwrap.wrap(prefix + text, width=text_w, subsequent_indent="  ")

    print(f"{colour}┌{'─' * (box_w - 2)}┐{reset}")
    for idx, ln in enumerate(lines):
        print(f"{colour}│{reset} ", end="")
        if idx == 0:
            print(f"{colour}{speaker_name}{reset}: ", end="", flush=True)
            chunk = ln[len(prefix):]
        else:
            chunk = ln
        if typewriter:
            for ch in chunk:
                print(ch, end="", flush=True)
                time.sleep(0.02)
        else:
            print(chunk, end="", flush=True)
        print(" " * (text_w - len(ln)), end="")
        print(f" {colour}│{reset}")
    print(f"{colour}└{'─' * (box_w - 2)}┘{reset}")

    if tts_engine and voice_id:
        try:
            tts_engine.setProperty("voice", voice_id)
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass 