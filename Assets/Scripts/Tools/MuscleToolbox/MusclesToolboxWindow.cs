#if UNITY_EDITOR
using System;
using MuscleToolbox;
using UnityEditor;
using UnityEngine;

namespace MusclesToolbox
{
    /// <summary>
    /// Main editor window that hosts multiple independent modules (tabs).
    /// Each module owns its GUI + optional SceneView logic.
    /// </summary>
    public sealed class MusclesToolboxWindow : EditorWindow
    {
        private enum Tab
        {
            SplineBrush = 0,
            ExpressionCapture = 1,
            CurveDefaults = 2
        }

        [SerializeField] private Tab _tab = Tab.SplineBrush;

        // Modules are serializable fields, so their settings persist without singletons.
        [SerializeField] private SplineBrushModule _splineBrush = new();
        [SerializeField] private ExpressionCaptureModule _expressionCapture = new();
        [SerializeField] private CurveDefaultsModule _curveDefaults = new();

        [MenuItem("Tools/Muscles/Muscle Toolbox")]
        public static void Open() => GetWindow<MusclesToolboxWindow>("Muscle Toolbox");

        private void OnEnable()
        {
            SceneView.duringSceneGui += OnSceneGUI;
            _splineBrush.OnEnable(this);
            _expressionCapture.OnEnable(this);
            _curveDefaults.OnEnable(this);
        }

        private void OnDisable()
        {
            SceneView.duringSceneGui -= OnSceneGUI;
            _splineBrush.OnDisable(this);
            _expressionCapture.OnDisable(this);
            _curveDefaults.OnDisable(this);
        }

        private void OnGUI()
        {
            DrawHeader();

            _tab = (Tab)GUILayout.Toolbar((int)_tab, new[]
            {
                "Spline Brush",
                "Expression Capture",
                "Curve Defaults"
            });

            EditorGUILayout.Space(10);

            switch (_tab)
            {
                case Tab.SplineBrush:
                    _splineBrush.OnGUI(this);
                    break;

                case Tab.ExpressionCapture:
                    _expressionCapture.OnGUI(this);
                    break;

                case Tab.CurveDefaults:
                    _curveDefaults.OnGUI(this);
                    break;
            }
        }

        private void OnSceneGUI(SceneView sceneView)
        {
            // Only active tab receives SceneView events.
            switch (_tab)
            {
                case Tab.SplineBrush:
                    _splineBrush.OnSceneGUI(this, sceneView);
                    break;

                case Tab.ExpressionCapture:
                    _expressionCapture.OnSceneGUI(this, sceneView);
                    break;

                case Tab.CurveDefaults:
                    _curveDefaults.OnSceneGUI(this, sceneView);
                    break;
            }
        }

        private static void DrawHeader()
        {
            EditorGUILayout.LabelField("Muscle Toolbox", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Modular: Brush / Capture / Defaults", EditorStyles.miniLabel);
            EditorGUILayout.Space(6);
        }
    }
}
#endif