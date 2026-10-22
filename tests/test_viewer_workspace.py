import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import view_solution as viewer
ROOT=Path(__file__).resolve().parents[1]

class ViewerWorkspace(unittest.TestCase):
    def test_axes_hidden_by_default_configuration(self):
        fig=plt.figure();ax=fig.add_subplot(projection='3d')
        viewer.configure_axes(ax,np.array([[0.,0.,0.],[1.,1.,1.]]),('x','y','z'),False)
        self.assertFalse(ax._axis3don)
        plt.close(fig)
    def test_small_gif(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'test.gif'
            p=subprocess.run([sys.executable,str(ROOT/'view_solution.py'),str(ROOT/'best_known/3d_81p_088871'),
                 '--gif',str(out),'--frames','4','--fps','4','--no-show'],
                 cwd=td,capture_output=True,text=True,timeout=25)
            self.assertEqual(p.returncode,0,p.stderr)
            with Image.open(out) as im:self.assertGreater(im.n_frames,1)
    def test_four_dimensional_still(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'test.png'
            p=subprocess.run([sys.executable,str(ROOT/'view_solution.py'),str(ROOT/'best_known/4d_61p_091314'),
                 '--save',str(out),'--no-show'],capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn('PCA projection',p.stdout)
            with Image.open(out) as im:self.assertGreater(min(im.size),50)
    def test_source_colors_loaded(self):
        _,x,c,meta=viewer.load_solution(ROOT/'best_known/3d_81p_088871')
        self.assertEqual(len(x),81);self.assertEqual(c.shape,(81,));self.assertEqual(meta['k'],4)

if __name__=='__main__':unittest.main()
