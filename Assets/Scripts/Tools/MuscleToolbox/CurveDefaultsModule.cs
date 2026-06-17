#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using MuscleToolbox;
using UnityEditor;
using UnityEngine;
using UnityEngine.Splines;

namespace MusclesToolbox
{
    /// <summary>
    /// Saves/restores curve defaults into CurveDefaultsAsset.
    /// Restore behavior: ONLY restores objects currently selected in Hierarchy
    /// (SplineContainer/LineRenderer on self or parents).
    /// </summary>
    [Serializable]
    public sealed class CurveDefaultsModule : IToolboxModule
    {
        [SerializeField] private CurveDefaultsAsset _defaultsAsset;

        public void OnEnable(EditorWindow host) { }
        public void OnDisable(EditorWindow host) { }
        public void OnSceneGUI(EditorWindow host, SceneView sceneView) { }

        public void OnGUI(EditorWindow host)
        {
            _defaultsAsset = (CurveDefaultsAsset)EditorGUILayout.ObjectField(
                "Defaults Asset",
                _defaultsAsset,
                typeof(CurveDefaultsAsset),
                false);

            EditorGUILayout.Space(8);

            if (_defaultsAsset == null)
            {
                EditorGUILayout.HelpBox(
                    "Create a CurveDefaultsAsset first:\n" +
                    "Create → Muscles → Curve Defaults Asset\n" +
                    "Then assign it here.",
                    MessageType.Info);
                return;
            }

            EditorGUILayout.HelpBox(
                "Select curve GameObjects in Hierarchy (SplineContainer and/or LineRenderer).\n" +
                "Save Default: stores current curve state into the asset.\n" +
                "Restore Default: applies stored state ONLY to selected objects.",
                MessageType.Info);

            EditorGUILayout.Space(8);

            if (GUILayout.Button("Save Default From Selection"))
                SaveFromSelection();

            if (GUILayout.Button("Restore Default To Selected"))
                RestoreToSelected();

            EditorGUILayout.Space(8);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Clear Stored Defaults"))
                {
                    Undo.RecordObject(_defaultsAsset, "Clear Curve Defaults");
                    _defaultsAsset.splineContainers.Clear();
                    _defaultsAsset.lineRenderers.Clear();
                    EditorUtility.SetDirty(_defaultsAsset);
                    AssetDatabase.SaveAssets();
                }

                if (GUILayout.Button("Ping Asset"))
                    EditorGUIUtility.PingObject(_defaultsAsset);
            }

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField(
                $"Stored: SplineContainers={_defaultsAsset.splineContainers.Count}, LineRenderers={_defaultsAsset.lineRenderers.Count}",
                EditorStyles.miniLabel);
        }

        private void SaveFromSelection()
        {
            var selected = Selection.gameObjects;
            if (selected == null || selected.Length == 0)
            {
                Debug.LogError("Nothing selected. Select SplineContainer/LineRenderer objects in Hierarchy.");
                return;
            }

            Undo.RecordObject(_defaultsAsset, "Save Curve Defaults");

            foreach (var go in selected)
            {
                var sc = go.GetComponent<SplineContainer>() ?? go.GetComponentInParent<SplineContainer>();
                if (sc != null)
                {
                    SaveSplineContainer(sc);
                    continue;
                }

                var lr = go.GetComponent<LineRenderer>() ?? go.GetComponentInParent<LineRenderer>();
                if (lr != null)
                {
                    SaveLineRenderer(lr);
                    continue;
                }

                Debug.LogWarning($"Skipping '{go.name}': no SplineContainer or LineRenderer found (self/parents).");
            }

            EditorUtility.SetDirty(_defaultsAsset);
            AssetDatabase.SaveAssets();
            Debug.Log("Saved defaults from selection.");
        }

        private void RestoreToSelected()
        {
            if (_defaultsAsset == null) return;

            var selected = Selection.gameObjects;
            if (selected == null || selected.Length == 0)
            {
                Debug.LogError("Nothing selected. Select SplineContainer/LineRenderer objects (or their children) to restore.");
                return;
            }

            var selectedPaths = new HashSet<string>();

            foreach (var go in selected)
            {
                if (go == null) continue;

                var sc = go.GetComponent<SplineContainer>() ?? go.GetComponentInParent<SplineContainer>();
                if (sc != null)
                {
                    selectedPaths.Add(GetHierarchyPath(sc.transform));
                    continue;
                }

                var lr = go.GetComponent<LineRenderer>() ?? go.GetComponentInParent<LineRenderer>();
                if (lr != null)
                {
                    selectedPaths.Add(GetHierarchyPath(lr.transform));
                }
            }

            if (selectedPaths.Count == 0)
            {
                Debug.LogError("Selection has no SplineContainer or LineRenderer (self/parents).");
                return;
            }

            int restoredSplineContainers = 0;
            int restoredLineRenderers = 0;

            foreach (var snap in _defaultsAsset.splineContainers)
            {
                if (!selectedPaths.Contains(snap.objectPath))
                    continue;

                var tr = FindByHierarchyPath(snap.objectPath);
                if (tr == null)
                {
                    Debug.LogWarning($"Cannot find object by path: {snap.objectPath}");
                    continue;
                }

                var sc = tr.GetComponent<SplineContainer>();
                if (sc == null)
                {
                    Debug.LogWarning($"Object found but no SplineContainer: {snap.objectPath}");
                    continue;
                }

                Undo.RecordObject(sc, "Restore Spline Defaults (Selected)");

                while (sc.Splines.Count < snap.splines.Count)
                    sc.AddSpline();
                while (sc.Splines.Count > snap.splines.Count)
                    sc.RemoveSplineAt(sc.Splines.Count - 1);

                for (int si = 0; si < snap.splines.Count; si++)
                {
                    var splineSnap = snap.splines[si];
                    var spline = sc.Splines[si];

                    spline.Clear();
                    for (int ki = 0; ki < splineSnap.knots.Length; ki++)
                    {
                        var k = splineSnap.knots[ki];
                        spline.Add(new BezierKnot(k.position, k.tangentIn, k.tangentOut, k.rotation));
                    }

                    spline.Closed = splineSnap.closed;
                }

                EditorUtility.SetDirty(sc);
                restoredSplineContainers++;
            }

            foreach (var snap in _defaultsAsset.lineRenderers)
            {
                if (!selectedPaths.Contains(snap.objectPath))
                    continue;

                var tr = FindByHierarchyPath(snap.objectPath);
                if (tr == null)
                {
                    Debug.LogWarning($"Cannot find object by path: {snap.objectPath}");
                    continue;
                }

                var lr = tr.GetComponent<LineRenderer>();
                if (lr == null)
                {
                    Debug.LogWarning($"Object found but no LineRenderer: {snap.objectPath}");
                    continue;
                }

                Undo.RecordObject(lr, "Restore LineRenderer Defaults (Selected)");

                lr.useWorldSpace = snap.useWorldSpace;
                lr.positionCount = snap.positions.Length;
                lr.SetPositions(snap.positions);

                EditorUtility.SetDirty(lr);
                restoredLineRenderers++;
            }

            Debug.Log($"Restore selected done. SplineContainers: {restoredSplineContainers}, LineRenderers: {restoredLineRenderers}");
        }

        private void SaveSplineContainer(SplineContainer sc)
        {
            string path = GetHierarchyPath(sc.transform);

            _defaultsAsset.splineContainers.RemoveAll(s => s.objectPath == path);

            var snap = new CurveDefaultsAsset.SplineContainerSnapshot
            {
                objectPath = path,
                splines = new List<CurveDefaultsAsset.SplineSnapshot>()
            };

            for (int si = 0; si < sc.Splines.Count; si++)
            {
                Spline spline = sc.Splines[si];

                var splineSnap = new CurveDefaultsAsset.SplineSnapshot
                {
                    closed = spline.Closed,
                    knots = new CurveDefaultsAsset.KnotSnapshot[spline.Count]
                };

                for (int ki = 0; ki < spline.Count; ki++)
                {
                    var knot = spline[ki];
                    splineSnap.knots[ki] = new CurveDefaultsAsset.KnotSnapshot
                    {
                        position = knot.Position,
                        tangentIn = knot.TangentIn,
                        tangentOut = knot.TangentOut,
                        rotation = knot.Rotation
                    };
                }

                snap.splines.Add(splineSnap);
            }

            _defaultsAsset.splineContainers.Add(snap);
        }

        private void SaveLineRenderer(LineRenderer lr)
        {
            string path = GetHierarchyPath(lr.transform);

            _defaultsAsset.lineRenderers.RemoveAll(s => s.objectPath == path);

            var snap = new CurveDefaultsAsset.LineRendererSnapshot
            {
                objectPath = path,
                useWorldSpace = lr.useWorldSpace,
                positions = new Vector3[lr.positionCount]
            };

            for (int i = 0; i < lr.positionCount; i++)
                snap.positions[i] = lr.GetPosition(i);

            _defaultsAsset.lineRenderers.Add(snap);
        }

        // ---------------- Helpers ----------------
        private static string GetHierarchyPath(Transform tr)
        {
            var stack = new Stack<string>();
            while (tr != null)
            {
                stack.Push(tr.name);
                tr = tr.parent;
            }
            return string.Join("/", stack);
        }

        private static Transform FindByHierarchyPath(string path)
        {
            var parts = path.Split('/');
            if (parts.Length == 0) return null;

            GameObject root = GameObject.Find(parts[0]);
            if (root == null) return null;

            Transform cur = root.transform;
            for (int i = 1; i < parts.Length; i++)
            {
                cur = cur.Find(parts[i]);
                if (cur == null) return null;
            }
            return cur;
        }
    }
}
#endif