#    Copyright 2023 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""General settings."""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Immutable settings for opentile.

    Construct with the desired values. To change the process-wide default, use
    ``set_default_settings(Settings(...))``. To apply settings to a block of
    code, use ``use_settings``.
    """

    ndpi_frame_cache: int = 128
    """Number of frames to cache for ndpi."""


_default_settings = Settings()
_active_settings: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "opentile_active_settings", default=None
)


def get_settings() -> Settings:
    """The settings in effect: those active in the current context (see
    ``use_settings``), or the process-wide default when none is active."""
    return _active_settings.get() or _default_settings


def set_default_settings(new_settings: Settings) -> None:
    """Replace the process-wide default settings.

    Parameters
    ----------
    new_settings: Settings
        The new process-wide default settings.
    """
    global _default_settings
    _default_settings = new_settings


@contextmanager
def use_settings(active: Settings | None = None) -> Iterator[Settings]:
    """Activate settings for the current context and yield the settings in effect.

    Use as ``with use_settings(Settings(...)) as settings:`` to apply settings to
    a block (and thread-pool tasks it submits that propagate the context), or
    ``with use_settings() as settings:`` to just read the settings in effect.

    Parameters
    ----------
    active: Settings | None = None
        Settings to activate for the current context. When None, nothing is
        activated and the settings currently in effect are yielded.

    Yields
    ------
    Settings
        The settings in effect within the context.
    """
    if active is None:
        yield get_settings()
        return
    token = _active_settings.set(active)
    try:
        yield active
    finally:
        _active_settings.reset(token)
