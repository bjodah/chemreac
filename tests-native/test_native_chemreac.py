# -*- coding: utf-8 -*-

import os
import subprocess



def test_native():
    p = subprocess.Popen(['make', '-B'], cwd=os.path.dirname(__file__))
    assert p.wait() == 0  # systems which have `make` have SUCCESS==0
