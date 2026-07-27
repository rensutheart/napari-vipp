"""Optional GPU runtimes.

Importing this package never imports an accelerator library or creates a
device context.  Provider modules perform their imports only when explicitly
probed or used.
"""

__all__: list[str] = []
