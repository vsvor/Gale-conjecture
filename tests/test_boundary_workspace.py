import json
from pathlib import Path
import shutil
import tempfile
import unittest
from verify_skeleton import verify,exact_skeleton
ROOT=Path(__file__).resolve().parents[1]

class BoundaryWorkspace(unittest.TestCase):
    def test_cube_skeleton_excludes_face_diagonals(self):
        rows=[(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
        edges,faces=exact_skeleton(rows)
        self.assertEqual(len(edges),12);self.assertEqual(faces,6)
        self.assertTrue(all(sum((a-b)**2 for a,b in zip(rows[u],rows[v]))==1 for u,v in edges))
    def test_complete_saved_cover_geometry(self):
        r=verify(ROOT/'best_known/cover49',check_lower=False)
        self.assertTrue(r['complete_skeleton_cover_verified']);self.assertEqual(r['source_points'],49)
    def test_missing_cover_edge_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'cover';shutil.copytree(ROOT/'best_known/cover49',target)
            data=json.loads((target/'cover.json').read_text());data['edges'].pop()
            (target/'cover.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'missing'):verify(target)

if __name__=='__main__':unittest.main()
