"""App package bootstrap."""

from __future__ import annotations

import inspect

try:
    from starlette.routing import Router as _StarletteRouter

    _router_init_sig = inspect.signature(_StarletteRouter.__init__)
    if "on_startup" not in _router_init_sig.parameters:
        _router_init = _StarletteRouter.__init__

        def _compat_router_init(
            self,
            *args,
            on_startup=None,
            on_shutdown=None,
            lifespan=None,
            **kwargs,
        ):
            _router_init(self, *args, **kwargs)
            self.on_startup = list(on_startup or [])
            self.on_shutdown = list(on_shutdown or [])
            self.lifespan_context = lifespan
            return None

        _StarletteRouter.__init__ = _compat_router_init
except Exception:
    pass
