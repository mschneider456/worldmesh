from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parents[1] / "method"
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

try:
    import trimesh
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    trimesh = None

if trimesh is not None:
    import fill_missing_windows
    from extract_objects import step3_position_merge


@unittest.skipUnless(trimesh is not None, "trimesh is required for scene GLB tests")
class FillMissingWindowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = Path(self.temp_dir.name)

    def _scene_json_path(self) -> Path:
        scene_json_path = self.work_dir / "scene.json"
        scene_data = {
            "metadata": {"wall_thickness": 0.15},
            "rooms": [
                {
                    "id": "room_a",
                    "floor_polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    "openings": [
                        {
                            "type": "window",
                            "wall_segment": [[0, 0], [4, 0]],
                            "position": 1.4,
                            "width": 1.2,
                            "height": 1.2,
                            "sill_height": 0.9,
                        }
                    ],
                }
            ],
        }
        scene_json_path.write_text(json.dumps(scene_data), encoding="utf-8")
        return scene_json_path

    def _window_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=[1.2, 0.15, 1.2])
        mesh.apply_translation([2.0, 0.075, 1.5])
        return mesh

    def _roundtrip_scene(self, scene: trimesh.Scene) -> trimesh.Scene:
        path = self.work_dir / "scene.glb"
        scene.export(path)
        loaded = trimesh.load(path)
        self.assertIsInstance(loaded, trimesh.Scene)
        return loaded

    def test_collect_filled_uses_graph_node_labels_after_glb_roundtrip(self) -> None:
        scene_json_path = self._scene_json_path()
        scene = trimesh.Scene()
        scene.add_geometry(self._window_mesh(), node_name="object_03_window")

        loaded = self._roundtrip_scene(scene)
        self.assertNotIn("object_03_window", loaded.geometry)
        self.assertIn("object_03_window", loaded.graph.nodes_geometry)

        filled, unfilled = fill_missing_windows.collect_filled_and_unfilled(
            loaded, scene_json_path
        )

        self.assertEqual(len(filled), 1)
        self.assertEqual(len(unfilled), 0)

    def test_collect_filled_falls_back_to_window_geometry_when_labels_are_generic(self) -> None:
        scene_json_path = self._scene_json_path()
        scene = trimesh.Scene()
        scene.add_geometry(
            self._window_mesh(),
            node_name="geometry_0",
            geom_name="geometry_0",
        )

        loaded = self._roundtrip_scene(scene)
        filled, unfilled = fill_missing_windows.collect_filled_and_unfilled(
            loaded, scene_json_path
        )

        self.assertEqual(len(filled), 1)
        self.assertEqual(len(unfilled), 0)

    def test_merge_meshes_into_scene_preserves_window_name_across_roundtrip(self) -> None:
        prior_scene = trimesh.Scene()
        prior_scene.add_geometry(self._window_mesh(), node_name="object_03_window")

        merged = step3_position_merge.merge_meshes_into_scene(prior_scene, [])
        loaded = self._roundtrip_scene(merged)

        self.assertIn("object_03_window", loaded.graph.nodes_geometry)
        self.assertIn("object_03_window", loaded.geometry)

    def test_clone_windows_for_unmatched_uses_room_id_in_clone_names(self) -> None:
        openings = step3_position_merge.load_room_openings(self._scene_json_path(), "room_a")
        source_opening = openings[0]
        source_mesh = self._window_mesh()

        target_opening = dict(source_opening)
        target_opening["center_2d"] = source_opening["center_2d"] + [1.0, 0.0]

        cloned = step3_position_merge.clone_windows_for_unmatched(
            [("object_03_window", source_mesh, source_opening)],
            [target_opening],
            [],
            0.15,
            str(self._scene_json_path()),
            "room_a",
            verbose=False,
        )

        self.assertEqual(len(cloned), 1)
        self.assertEqual(cloned[0][0], "cloned_window_room_a_00")
