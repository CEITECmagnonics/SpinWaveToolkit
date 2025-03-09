def _can_import():
    try:
        import SpinWaveToolkit
        return True
    except ImportError:
        return False


class TestClass:
    def test_import_swt(self):
        assert _can_import()
