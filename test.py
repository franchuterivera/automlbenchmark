import os
import mock

class Dummy:
    def __init__():
        pass


@mock.patch("os.listdir", spec=Dummy)
def test1(what):
    assert isinstance(what, Dummy)
