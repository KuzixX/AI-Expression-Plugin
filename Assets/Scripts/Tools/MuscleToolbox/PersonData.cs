using System;
using System.Collections.Generic;
using FaceRig.Legacy;
using UnityEngine;

[CreateAssetMenu(menuName = "Muscles/Person Data")]
public class PersonData : ScriptableObject
{
    public string personId;

    [Tooltip("Rest lengths per muscle/curve (same order as your selection order).")]
    public float[] restLengths;

    [Tooltip("Gain per muscle: activation = clamp01(-strain * gain). Optional. If empty, window Default Gain is used.")]
    public float[] gains;

    public List<ExpressionSample> samples = new();

    [Serializable]
    public class ExpressionSample
    {
        public string expressionName; // e.g. "Smile"
        public float[] lengths;       // measured lengths at capture time
        public float[] strain;        // (L - Lrest) / Lrest
        public RigActivationsContainer[] activation;    // clamp01(-strain * gain)
        public string[] curveNames;   // debug: which curves were used (and their order)
        public long capturedUtcTicks;
    }

    public ExpressionSample FindSample(string name)
        => samples.Find(s => string.Equals(s.expressionName, name, StringComparison.OrdinalIgnoreCase));
}