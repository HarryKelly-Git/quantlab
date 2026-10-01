"""Options layer: indicative-quote evaluation of stock vs defined-risk option structures, and a
separate PAPER options book (OPT), gated OFF by default. See docs/OPTIONS.md.

Deliberately import-free: ``pipeline/runner.py`` imports ``quantlab.options.book`` and must not pull
in the rest of the layer (or create an import cycle with ``execution/``).
"""
