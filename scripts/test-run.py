#!/usr/bin/env python3
"""Interactive test runner.

Launches `llama cli -p yuki --no-tts`, auto-picks model 4 when the picker
appears, then hands the terminal over to you for prompt-by-prompt testing.
Every byte (your input AND Yuki's replies, including 💗 patience logs and
any tracebacks) is mirrored to yuki-run.log.

Quit with Ctrl-D or the CLI's own exit.
"""
from __future__ import annotations
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "yuki-run.log"
CLI = ["llama", "cli", "-p", "yuki", "--no-tts"]
MODEL_CHOICE = b"4\n"
TRIGGER = b"Select model number:"


def main() -> int:
    log = open(LOG, "wb")
    log.write(b"\n" + b"=" * 70 + b"\n")
    log.write(f"  session started, model slot 4, persona yuki\n".encode())
    log.write(b"=" * 70 + b"\n\n")
    log.flush()

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(CLI[0], CLI)

    stdin_fd = sys.stdin.fileno()

    def propagate_size(*_):
        """Copy the real terminal's size into the child pty."""
        try:
            size = fcntl.ioctl(stdin_fd, termios.TIOCGWINSZ, b"\x00" * 8)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
        except OSError:
            pass

    propagate_size()
    signal.signal(signal.SIGWINCH, propagate_size)
    try:
        saved = termios.tcgetattr(stdin_fd)
    except termios.error:
        saved = None

    if saved is not None:
        tty.setcbreak(stdin_fd)  # cbreak (not raw) so Ctrl-C still reaches the child

    sent_choice = False
    seen = b""

    def pump(n=4096):
        try:
            return os.read(fd, n)
        except OSError:
            return b""

    try:
        while True:
            r, _, _ = select.select([fd, stdin_fd], [], [])
            if fd in r:
                data = pump()
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
                log.write(data)
                log.flush()
                if not sent_choice:
                    seen += data
                    if TRIGGER in seen:
                        os.write(fd, MODEL_CHOICE)
                        sent_choice = True
                        seen = b""  # release memory, no longer needed
            if stdin_fd in r:
                data = os.read(stdin_fd, 1024)
                if not data:
                    os.close(fd)
                    break
                os.write(fd, data)
    finally:
        if saved is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, saved)
        log.close()

    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        status = 0
    print(f"\n(log saved to {LOG.relative_to(ROOT)})")
    return os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
