#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using FaceRig.Legacy;
using UnityEditor;
using UnityEngine;
using UnityEngine.Splines;

namespace MusclesToolbox
{
    /// <summary>
    /// Captures curve length from selected SplineContainer/LineRenderer objects into PersonData samples.
    /// - Neutral capture writes restLengths only (no sample created).
    /// - Expression capture writes lengths/strain/activation + timestamp.
    /// </summary>
    [Serializable]
    public sealed class ExpressionCaptureModule : IToolboxModule
    {
        [SerializeField] private PersonData _person;
        [SerializeField] private string _expressionName = "Smile";
        
        [SerializeField] private float _defaultGain = 10f;

        [SerializeField] private int _splineSamples = 64;

        public void OnEnable(EditorWindow host) { }
        public void OnDisable(EditorWindow host) { }
        public void OnSceneGUI(EditorWindow host, SceneView sceneView) { }

        public void OnGUI(EditorWindow host)
        {
            _person = (PersonData)EditorGUILayout.ObjectField("Person Data", _person, typeof(PersonData), false);
            _expressionName = EditorGUILayout.TextField("Expression Name", _expressionName);

            EditorGUILayout.Space(6);
            _defaultGain = EditorGUILayout.FloatField("Default Gain (if Person.gains empty)", _defaultGain);

            EditorGUILayout.Space(6);
            _splineSamples = EditorGUILayout.IntSlider("Spline Samples (length)", _splineSamples, 16, 256);

            EditorGUILayout.Space(10);

            using (new EditorGUI.DisabledScope(_person == null))
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Capture NEUTRAL (RestLengths)"))
                        CaptureNeutral();

                    if (GUILayout.Button("Capture EXPRESSION Sample"))
                        CaptureExpression();
                }
            }

            EditorGUILayout.HelpBox(
                "Workflow:\n" +
                "1) Select curves (SplineContainer/LineRenderer) in Hierarchy\n" +
                "2) Capture NEUTRAL once (writes restLengths)\n" +
                "3) Pose face and Capture EXPRESSION sample (writes lengths/strain/activation)\n\n" +
                "Tip: selecting a child object is ok — tool searches in parents.",
                MessageType.Info);
        }

private void CaptureNeutral()
{
    if (!TryCollectSelection(out var namesArr, out var lenArr))
        return;

    int m = lenArr.Length;

    Undo.RecordObject(_person, "Capture Neutral");
    EnsureArray(ref _person.restLengths, m);

    for (int i = 0; i < m; i++)
        _person.restLengths[i] = Mathf.Max(1e-6f, lenArr[i]);

    EditorUtility.SetDirty(_person);
    AssetDatabase.SaveAssets();

    Debug.Log($"Captured NEUTRAL for '{_person.name}': wrote restLengths for {m} curves (no sample created).");
    Selection.activeObject = _person;
}

private bool TryCollectSelection(out string[] namesArr, out float[] lenArr)
{
    namesArr = null;
    lenArr = null;

    if (_person == null)
    {
        Debug.LogError("PersonData is not set.");
        return false;
    }

    var selected = Selection.gameObjects;
    if (selected == null || selected.Length == 0)
    {
        Debug.LogError("Nothing selected. Select curve GameObjects in Hierarchy.");
        return false;
    }

    var curveNames = new List<string>();
    var lengths = new List<float>();

    foreach (var go in selected)
    {
        if (TryGetCurveLength(go, _splineSamples, out float length, out string usedName))
        {
            curveNames.Add(usedName);
            lengths.Add(length);
        }
        else
        {
            Debug.LogWarning($"Skipping '{go.name}': no SplineContainer/LineRenderer found (self/parents).");
        }
    }

    if (lengths.Count == 0)
    {
        Debug.LogError("No supported curves found in selection.");
        return false;
    }

    namesArr = curveNames.ToArray();
    lenArr = lengths.ToArray();
    return true;
}

private void CaptureExpression()
{
    if (!TryCollectSelection(out var namesArr, out var lenArr))
        return;

    int m = lenArr.Length;

    Undo.RecordObject(_person, "Capture Expression");

    // Ensure arrays
    EnsureArray(ref _person.restLengths, m);

    bool hasGains = _person.gains != null && _person.gains.Length == m;

    float[] strain = new float[m];
    RigActivationsContainer[] activation = new RigActivationsContainer[m];

    for (int i = 0; i < m; i++)
    {
        float lRest = _person.restLengths[i];

        if (lRest <= 1e-6f)
        {
            Debug.LogWarning($"restLengths[{i}] not set. Capture NEUTRAL first. Curve='{namesArr[i]}'");
            lRest = Mathf.Max(1e-6f, lenArr[i]);
        }

        float eps = (lenArr[i] - lRest) / lRest;
        strain[i] = eps;

        float gain = hasGains ? _person.gains[i] : _defaultGain;

        // ВАЖНО: создать контейнер, иначе activation[i] == null
        activation[i] = new RigActivationsContainer
        {
            // jointTargetTag = ???  (если надо — заполни тут)
            activation = Mathf.Clamp01((-eps) * gain)
        };
    }

    var sample = _person.FindSample(_expressionName);
    if (sample == null)
    {
        sample = new PersonData.ExpressionSample { expressionName = _expressionName };
        _person.samples.Add(sample);
    }

    sample.lengths = lenArr;
    sample.strain = strain;
    sample.activation = activation;
    sample.curveNames = namesArr;
    sample.capturedUtcTicks = DateTime.UtcNow.Ticks;

    EditorUtility.SetDirty(_person);
    AssetDatabase.SaveAssets();

    Debug.Log($"Captured '{_expressionName}' for '{_person.name}' with {m} curves.");
    Selection.activeObject = _person;
}

        private static void EnsureArray(ref float[] arr, int len)
        {
            if (arr == null || arr.Length != len)
                arr = new float[len];
        }

        private static bool TryGetCurveLength(GameObject go, int samples, out float length, out string usedName)
        {
            length = 0f;
            usedName = go.name;

            // SplineContainer (self/parent)
            var sc = go.GetComponent<SplineContainer>() ?? go.GetComponentInParent<SplineContainer>();
            if (sc != null && sc.Splines != null && sc.Splines.Count > 0)
            {
                var spline = sc.Splines[0];
                length = ApproximateSplineLength(sc.transform, spline, samples);
                usedName = sc.gameObject.name + " (SplineContainer)";
                return length > 0f;
            }

            // LineRenderer (self/parent)
            var lr = go.GetComponent<LineRenderer>() ?? go.GetComponentInParent<LineRenderer>();
            if (lr != null && lr.positionCount >= 2)
            {
                Vector3 prev = lr.GetPosition(0);
                prev = lr.useWorldSpace ? prev : lr.transform.TransformPoint(prev);

                float sum = 0f;
                for (int i = 1; i < lr.positionCount; i++)
                {
                    Vector3 p = lr.GetPosition(i);
                    p = lr.useWorldSpace ? p : lr.transform.TransformPoint(p);
                    sum += Vector3.Distance(prev, p);
                    prev = p;
                }

                length = sum;
                usedName = lr.gameObject.name + " (LineRenderer)";
                return length > 0f;
            }

            return false;
        }

        private static float ApproximateSplineLength(Transform tr, Spline spline, int samples)
        {
            float sum = 0f;

            Vector3 prev = tr.TransformPoint(spline.EvaluatePosition(0f));
            for (int i = 1; i <= samples; i++)
            {
                float t = (float)i / samples;
                Vector3 p = tr.TransformPoint(spline.EvaluatePosition(t));
                sum += Vector3.Distance(prev, p);
                prev = p;
            }

            return sum;
        }
    }
}
#endif