"""load_droid must not hide a broken tensorflow_datasets import (found on Colab: protobuf 5.x with a tensorflow-metadata generated for 6.31.1)."""

from __future__ import annotations

import importlib.abc
import sys
import types

import pytest

from adjointrwm.data import load_droid


def _fake_tensorflow():
    module = types.ModuleType("tensorflow")
    module.config = types.SimpleNamespace(set_visible_devices=lambda *args: None)
    return module


def _fake_tfds(monkeypatch, *, with_load: bool):
    module = types.ModuleType("tensorflow_datasets")
    module.__path__ = []
    if with_load:
        module.load = lambda name, **kwargs: (f"dataset:{name}", "info")
    monkeypatch.setitem(sys.modules, "tensorflow", _fake_tensorflow())
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", module)
    return module


class _Boom(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "tensorflow_datasets.public_api":
            raise ValueError("swallowed import error")
        return None


def test_the_real_import_error_is_raised_when_load_is_missing(monkeypatch):
    _fake_tfds(monkeypatch, with_load=False)
    monkeypatch.setattr(sys, "meta_path", [_Boom(), *sys.meta_path])
    with pytest.raises(ValueError, match="swallowed import error"):
        load_droid()


def test_a_module_without_load_and_without_an_error_still_fails_loudly(monkeypatch):
    _fake_tfds(monkeypatch, with_load=False)
    monkeypatch.setitem(sys.modules, "tensorflow_datasets.public_api", types.ModuleType("tensorflow_datasets.public_api"))
    with pytest.raises(RuntimeError, match="without its public API"):
        load_droid()


def test_a_healthy_tensorflow_datasets_is_used_unchanged(monkeypatch):
    _fake_tfds(monkeypatch, with_load=True)
    assert load_droid("droid_100", "gs://somewhere") == ("dataset:droid_100", "info")
