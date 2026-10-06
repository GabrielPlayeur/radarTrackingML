"""Test de fumée : les cinq sous-paquets de src/ sont importables (structure du ROADMAP)."""

import importlib

import pytest


@pytest.mark.parametrize("package", ["radar", "sim", "tracking", "learned", "metrics"])
def test_package_importable(package):
    """Chaque sous-paquet défini dans « Structure du dépôt » doit s'importer."""
    assert importlib.import_module(package) is not None
