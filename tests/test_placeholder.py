import importlib


def test_package_importable():
    module = importlib.import_module("football_video_analyser")
    assert hasattr(module, "__all__")
