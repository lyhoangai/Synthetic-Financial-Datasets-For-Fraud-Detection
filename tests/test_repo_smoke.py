from importlib import import_module


def test_core_modules_can_be_imported():
    import_module("src.config")
    import_module("src.data")
    import_module("src.features")
    import_module("src.models")
    import_module("src.rules")
