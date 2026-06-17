#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using MusclesToolbox;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Splines;

namespace MuscleToolbox
{
    /// <summary>
    /// Spline Brush module:
    /// - Draw mode: click points on collider, click first point to close.
    /// - Slide mode: move knots of ALL selected SplineContainers along the surface inside a radius.
    ///   Each moved knot is re-projected onto surface using raycast along normal.
    /// </summary>
    [Serializable]
    public sealed class SplineBrushModule : IToolboxModule
    {
        private enum BrushMode
        {
            Draw = 0,
            Slide = 1
        }
        // --- Tool activation ---
        [SerializeField] private bool _editEnabled = false;

        // --- Cursor marker ---
        [SerializeField] private bool _showCursorMarker = true;
        [SerializeField] private float _cursorMarkerRadius = 0.025f; // meters (world)
        [SerializeField] private BrushMode _mode = BrushMode.Draw;

        // --- Shared settings ---
        [SerializeField] private LayerMask _paintMask = ~0;
        [SerializeField] private Transform _explicitTarget;

        // --- Draw settings ---
        [SerializeField] private TangentMode _tangentMode = TangentMode.AutoSmooth;
        [SerializeField] private float _closeClickRadius = 0.05f; // meters (world)
        [SerializeField] private string _splineObjectName = "ClickSpline";

        // --- Slide settings ---
        [SerializeField] private float _slideRadius = 0.25f;     // meters (world)
        [SerializeField] private float _slideFalloffPow = 1.0f;  // 1=linear, 2=stronger center
        [SerializeField] private bool _showSlideRadius = true;

        // --- Draw runtime ---
        private SplineContainer _brushContainer;
        private readonly List<Vector3> _brushWorldPoints = new();
        private bool _brushClosed;

        // --- Slide runtime ---
        [Serializable]
        private sealed class SlideTarget
        {
            public SplineContainer SplineContainer;
            public int[] KnotIndices;
            public Vector3[] KnotWorldStart;
            public float[] Weights;
        }

        private readonly List<SlideTarget> _slideTargets = new();
        private bool _isSliding;
        private Vector3 _slideStartHitPoint;
        private Vector3 _slideHitNormal;

        private bool _hasLastHit;
        private Vector3 _lastHitPoint;
        private Vector3 _lastHitNormal;

        public void OnEnable(EditorWindow host) { }
        public void OnDisable(EditorWindow host) { }

        public void OnGUI(EditorWindow host)
        {
            _mode = (BrushMode)EditorGUILayout.EnumPopup("Brush Mode", _mode);
            
            EditorGUILayout.Space(6);

            using (new EditorGUILayout.HorizontalScope())
            {
                // Toggle-style button
                var wasEnabled = _editEnabled;
                _editEnabled = GUILayout.Toggle(_editEnabled, "Edit", "Button", GUILayout.Height(28));

                if (_editEnabled != wasEnabled)
                {
                    // On disable: cancel any in-progress actions to avoid "stuck" states
                    if (!_editEnabled)
                    {
                        _isSliding = false;
                        _slideTargets.Clear();
                        _brushWorldPoints.Clear();
                        _brushClosed = false;
                    }

                    SceneView.RepaintAll();
                }

                GUILayout.FlexibleSpace();

                _showCursorMarker = GUILayout.Toggle(_showCursorMarker, "Cursor Marker", "Button", GUILayout.Height(28));
            }

            EditorGUILayout.HelpBox(
                _editEnabled
                    ? "EDIT ENABLED: tool captures clicks in SceneView."
                    : "Edit is OFF: SceneView works normally (selection/move/etc).",
                _editEnabled ? MessageType.Info : MessageType.None);

            if (_mode == BrushMode.Slide)
            {
                _slideRadius = EditorGUILayout.Slider("Slide Radius", _slideRadius, 0.01f, 2.0f);
                _slideFalloffPow = EditorGUILayout.Slider("Falloff Power", _slideFalloffPow, 0.25f, 4.0f);
                _showSlideRadius = EditorGUILayout.ToggleLeft("Show Radius In SceneView", _showSlideRadius);

                EditorGUILayout.HelpBox(
                    "Slide mode (GROUP):\n" +
                    "• Select multiple SplineContainer objects (or their children)\n" +
                    "• LMB down near spline: picks knots inside radius (for ALL selected)\n" +
                    "• Drag: moves + projects knots back to surface\n" +
                    "• Mouse up: release\n" +
                    "• Esc / RMB: cancel",
                    MessageType.Info);
            }

            EditorGUILayout.HelpBox(
                "Draw mode:\n" +
                "• LMB on collider: add point\n" +
                "• Click near first point: close\n" +
                "• Output SplineContainer is created as CHILD of clicked object (or Explicit Target)\n" +
                "• Esc / RMB: cancel current stroke",
                MessageType.Info);

            _paintMask = LayerMaskField("Paint Mask", _paintMask);
            _explicitTarget = (Transform)EditorGUILayout.ObjectField("Explicit Target (optional)", _explicitTarget, typeof(Transform), true);

            EditorGUILayout.Space(8);

            _splineObjectName = EditorGUILayout.TextField("Spline Object Name", _splineObjectName);
            _tangentMode = (TangentMode)EditorGUILayout.EnumPopup("Tangent Mode", _tangentMode);
            _closeClickRadius = EditorGUILayout.Slider("Close Click Radius", _closeClickRadius, 0.005f, 0.5f);

            EditorGUILayout.Space(10);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("New Stroke (Clear Points)"))
                {
                    _brushWorldPoints.Clear();
                    _brushClosed = false;
                }

                if (GUILayout.Button("Force Create New Named Object"))
                {
                    _brushContainer = null;
                    _brushWorldPoints.Clear();
                    _brushClosed = false;
                }

                if (GUILayout.Button("Delete Output Object"))
                {
                    if (_brushContainer != null)
                    {
                        Undo.DestroyObjectImmediate(_brushContainer.gameObject);
                        _brushContainer = null;
                    }

                    _brushWorldPoints.Clear();
                    _brushClosed = false;
                }
            }

            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField($"Draw Points: {_brushWorldPoints.Count}   Closed: {_brushClosed}", EditorStyles.miniLabel);
            if (_brushContainer != null)
                EditorGUILayout.LabelField($"Brush Output: {_brushContainer.name}", EditorStyles.miniLabel);
        }

        public void OnSceneGUI(EditorWindow host, SceneView sceneView)
        {
            // If tool not armed — do nothing and do NOT block selection.
            if (!_editEnabled)
                return;

            // Prevent scene selection while tool is active.
            if (Event.current.type == EventType.Layout)
                HandleUtility.AddDefaultControl(GUIUtility.GetControlID(FocusType.Passive));

            UpdateLastSurfaceHit(Event.current);

            if (_mode == BrushMode.Draw)
                HandleDrawInput(Event.current);
            else
                HandleSlideInput(Event.current);

            // Visual feedback that tool is active
            DrawCursorMarkerGizmo();
            DrawSlideRadiusGizmo();
        }

        private void DrawCursorMarkerGizmo()
        {
            if (!_showCursorMarker) return;

            // show marker either at last hit (hover), or at slide start during drag
            Vector3 p;
            Vector3 n;

            if (_isSliding)
            {
                p = _slideStartHitPoint;
                n = _slideHitNormal;
            }
            else if (_hasLastHit)
            {
                p = _lastHitPoint;
                n = _lastHitNormal;
            }
            else
            {
                return;
            }

            Handles.zTest = UnityEngine.Rendering.CompareFunction.LessEqual;

            // Small ring + dot to clearly show "active"
            Handles.DrawWireDisc(p, n, _cursorMarkerRadius);
            Handles.DrawSolidDisc(p, n, _cursorMarkerRadius * 0.35f);
        }
        
        // ------------------------------------------------------------------
        // Draw mode
        // ------------------------------------------------------------------
        private void HandleDrawInput(Event e)
        {
            if ((e.type == EventType.KeyDown && e.keyCode == KeyCode.Escape) ||
                (e.type == EventType.MouseDown && e.button == 1))
            {
                _brushWorldPoints.Clear();
                _brushClosed = false;
                e.Use();
                return;
            }

            if (e.type == EventType.MouseDown && e.button == 0 && !_brushClosed)
            {
                var ray = HandleUtility.GUIPointToWorldRay(e.mousePosition);

                if (!Physics.Raycast(ray, out var hit, 5000f, _paintMask, QueryTriggerInteraction.Ignore))
                    return;

                if (!IsHitAllowedByExplicitTarget(hit.transform))
                    return;

                var parentForSpline = (_explicitTarget != null) ? _explicitTarget : hit.transform;
                EnsureBrushContainerExists(parentForSpline);

                Vector3 p = hit.point;

                if (_brushWorldPoints.Count >= 3 && Vector3.Distance(p, _brushWorldPoints[0]) <= _closeClickRadius)
                {
                    _brushClosed = true;
                    WriteBrushToSpline(closed: true);
                    e.Use();
                    return;
                }

                _brushWorldPoints.Add(p);
                WriteBrushToSpline(closed: false);

                e.Use();
            }
        }

        private void EnsureBrushContainerExists(Transform parent)
        {
            if (_brushContainer != null)
            {
                if (_brushContainer.transform.parent != parent)
                {
                    Undo.SetTransformParent(_brushContainer.transform, parent, "Reparent SplineContainer");
                    ResetLocalTransform(_brushContainer.transform);
                }

                return;
            }

            string name = string.IsNullOrWhiteSpace(_splineObjectName) ? "ClickSpline" : _splineObjectName.Trim();

            var go = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(go, "Create ClickSpline");
            Undo.SetTransformParent(go.transform, parent, "Parent ClickSpline");
            ResetLocalTransform(go.transform);

            _brushContainer = go.AddComponent<SplineContainer>();
            _brushContainer.AddSpline();

            Selection.activeGameObject = go;
            EditorUtility.SetDirty(_brushContainer);
        }

        private void WriteBrushToSpline(bool closed)
        {
            if (_brushContainer == null) return;

            Undo.RecordObject(_brushContainer, "Write Spline");

            var spline = _brushContainer.Spline;
            spline.Clear();

            var tr = _brushContainer.transform;

            for (int i = 0; i < _brushWorldPoints.Count; i++)
            {
                float3 pL = (float3)tr.InverseTransformPoint(_brushWorldPoints[i]);
                spline.Add(new BezierKnot(pL), _tangentMode);
            }

            spline.Closed = closed;

            EditorUtility.SetDirty(_brushContainer);
        }

        // ------------------------------------------------------------------
        // Slide mode (GROUP)
        // ------------------------------------------------------------------
        private void HandleSlideInput(Event e)
        {
            // Cancel
            if ((e.type == EventType.KeyDown && e.keyCode == KeyCode.Escape) ||
                (e.type == EventType.MouseDown && e.button == 1))
            {
                CancelSlide();
                e.Use();
                return;
            }

            // Start slide
            if (e.type == EventType.MouseDown && e.button == 0 && !_isSliding)
            {
                if (!TryRaycastSurface(e.mousePosition, out var hit))
                    return;

                if (!IsHitAllowedByExplicitTarget(hit.transform))
                    return;

                var group = GetSelectedSplineContainersUnique();

                // Optional fallbacks
                if (group.Count == 0)
                {
                    if (_brushContainer != null) group.Add(_brushContainer);
                    else
                    {
                        var scHit = hit.transform.GetComponent<SplineContainer>() ?? hit.transform.GetComponentInParent<SplineContainer>();
                        if (scHit != null) group.Add(scHit);
                    }
                }

                if (group.Count == 0)
                {
                    Debug.LogWarning("Slide: Select one or more objects with SplineContainer (or their children).");
                    return;
                }

                if (!TryBeginSlideGroup(group, hit.point, hit.normal))
                    return;

                e.Use();
                return;
            }

            // Drag
            if (e.type == EventType.MouseDrag && e.button == 0 && _isSliding)
            {
                if (!TryRaycastSurface(e.mousePosition, out var hit))
                    return;

                if (!IsHitAllowedByExplicitTarget(hit.transform))
                    return;

                ApplySlideToHit(hit.point, _slideHitNormal);
                e.Use();
                return;
            }

            // End
            if (e.type == EventType.MouseUp && e.button == 0 && _isSliding)
            {
                EndSlide();
                e.Use();
            }
        }

        private void CancelSlide()
        {
            _isSliding = false;
            _slideTargets.Clear();
        }

        private void EndSlide()
        {
            _isSliding = false;
            _slideTargets.Clear();
        }

        private bool TryBeginSlideGroup(List<SplineContainer> containers, Vector3 hitPoint, Vector3 hitNormal)
        {
            _slideTargets.Clear();

            _slideStartHitPoint = hitPoint;
            _slideHitNormal = (hitNormal.sqrMagnitude > 1e-6f) ? hitNormal.normalized : Vector3.up;

            foreach (var sc in containers)
            {
                if (sc == null) continue;

                var spline = sc.Spline; // first spline for simplicity
                if (spline == null || spline.Count == 0)
                    continue;

                var tr = sc.transform;

                var indices = new List<int>();
                var starts = new List<Vector3>();
                var weights = new List<float>();

                for (int i = 0; i < spline.Count; i++)
                {
                    Vector3 w = tr.TransformPoint((Vector3)spline[i].Position);
                    float d = Vector3.Distance(w, hitPoint);

                    if (d > _slideRadius)
                        continue;

                    float t = 1f - Mathf.Clamp01(d / Mathf.Max(1e-6f, _slideRadius));
                    float wgt = Mathf.Pow(t, _slideFalloffPow);

                    indices.Add(i);
                    starts.Add(w);
                    weights.Add(wgt);
                }

                if (indices.Count == 0)
                    continue;

                Undo.RecordObject(sc, "Slide Spline Knots (Group)");

                _slideTargets.Add(new SlideTarget
                {
                    SplineContainer = sc,
                    KnotIndices = indices.ToArray(),
                    KnotWorldStart = starts.ToArray(),
                    Weights = weights.ToArray()
                });
            }

            if (_slideTargets.Count == 0)
                return false;

            _isSliding = true;
            return true;
        }

        private void ApplySlideToHit(Vector3 newHitPoint, Vector3 stableNormal)
        {
            if (_slideTargets.Count == 0)
                return;

            Vector3 delta = newHitPoint - _slideStartHitPoint;

            foreach (var t in _slideTargets)
            {
                if (t?.SplineContainer == null) continue;

                var sc = t.SplineContainer;
                var spline = sc.Spline;
                var tr = sc.transform;

                for (int n = 0; n < t.KnotIndices.Length; n++)
                {
                    int i = t.KnotIndices[n];
                    float wgt = t.Weights[n];

                    Vector3 target = t.KnotWorldStart[n] + delta * wgt;

                    Vector3 projected = ProjectPointToSurface(target, stableNormal);

                    var old = spline[i];
                    float3 localPos = (float3)tr.InverseTransformPoint(projected);
                    spline[i] = new BezierKnot(localPos, old.TangentIn, old.TangentOut, old.Rotation);
                }

                EditorUtility.SetDirty(sc);
            }
        }

        private Vector3 ProjectPointToSurface(Vector3 worldPoint, Vector3 normal)
        {
            const float castHalf = 1.0f;

            Vector3 origin = worldPoint + normal * castHalf;
            if (Physics.Raycast(origin, -normal, out var hit, castHalf * 2f, _paintMask, QueryTriggerInteraction.Ignore))
                return hit.point;

            return worldPoint;
        }

        private static List<SplineContainer> GetSelectedSplineContainersUnique()
        {
            var result = new List<SplineContainer>();
            var set = new HashSet<SplineContainer>();

            var selected = Selection.gameObjects;
            if (selected == null) return result;

            foreach (var go in selected)
            {
                if (go == null) continue;

                var sc = go.GetComponent<SplineContainer>() ?? go.GetComponentInParent<SplineContainer>();
                if (sc != null && set.Add(sc))
                    result.Add(sc);
            }

            return result;
        }

        // ------------------------------------------------------------------
        // Scene gizmo: slide radius
        // ------------------------------------------------------------------
        private void UpdateLastSurfaceHit(Event e)
        {
            if (e == null) return;

            if (e.type != EventType.MouseMove &&
                e.type != EventType.MouseDrag &&
                e.type != EventType.Repaint &&
                e.type != EventType.Layout)
                return;

            if (!TryRaycastSurface(e.mousePosition, out var hit))
            {
                _hasLastHit = false;
                return;
            }

            if (!IsHitAllowedByExplicitTarget(hit.transform))
            {
                _hasLastHit = false;
                return;
            }

            _hasLastHit = true;
            _lastHitPoint = hit.point;
            _lastHitNormal = (hit.normal.sqrMagnitude > 1e-6f) ? hit.normal.normalized : Vector3.up;

            SceneView.RepaintAll();
        }

        private void DrawSlideRadiusGizmo()
        {
            if (_mode != BrushMode.Slide) return;
            if (!_showSlideRadius) return;

            Vector3 p, n;

            if (_isSliding)
            {
                p = _slideStartHitPoint;
                n = _slideHitNormal;
            }
            else if (_hasLastHit)
            {
                p = _lastHitPoint;
                n = _lastHitNormal;
            }
            else
            {
                return;
            }

            Handles.zTest = UnityEngine.Rendering.CompareFunction.LessEqual;
            Handles.DrawWireDisc(p, n, _slideRadius);
        }

        // ------------------------------------------------------------------
        // Utility
        // ------------------------------------------------------------------
        private bool TryRaycastSurface(Vector2 guiMousePos, out RaycastHit hit)
        {
            var ray = HandleUtility.GUIPointToWorldRay(guiMousePos);
            return Physics.Raycast(ray, out hit, 5000f, _paintMask, QueryTriggerInteraction.Ignore);
        }

        private bool IsHitAllowedByExplicitTarget(Transform hitTransform)
        {
            if (_explicitTarget == null)
                return true;

            return hitTransform == _explicitTarget || hitTransform.IsChildOf(_explicitTarget);
        }

        private static void ResetLocalTransform(Transform t)
        {
            t.localPosition = Vector3.zero;
            t.localRotation = Quaternion.identity;
            t.localScale = Vector3.one;
        }

        private static LayerMask LayerMaskField(string label, LayerMask selected)
        {
            var layers = new List<string>();
            var layerNumbers = new List<int>();

            for (int i = 0; i < 32; i++)
            {
                string n = LayerMask.LayerToName(i);
                if (!string.IsNullOrEmpty(n))
                {
                    layers.Add(n);
                    layerNumbers.Add(i);
                }
            }

            int maskWithoutEmpty = 0;
            for (int i = 0; i < layerNumbers.Count; i++)
                if (((1 << layerNumbers[i]) & selected.value) != 0)
                    maskWithoutEmpty |= 1 << i;

            maskWithoutEmpty = EditorGUILayout.MaskField(label, maskWithoutEmpty, layers.ToArray());

            int mask = 0;
            for (int i = 0; i < layerNumbers.Count; i++)
                if ((maskWithoutEmpty & (1 << i)) != 0)
                    mask |= 1 << layerNumbers[i];

            selected.value = mask;
            return selected;
        }
    }
}
#endif