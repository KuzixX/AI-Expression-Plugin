#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace MuscleToolbox
{
    [CreateAssetMenu(menuName = "Muscles/Curve Defaults Asset", fileName = "CurveDefaultsAsset")]
    public class CurveDefaultsAsset : ScriptableObject
    {
        [Serializable]
        public struct KnotSnapshot
        {
            public float3 position;
            public float3 tangentIn;
            public float3 tangentOut;
            public quaternion rotation;
        }

        [Serializable]
        public class SplineSnapshot
        {
            public bool closed;
            public KnotSnapshot[] knots;
        }

        [Serializable]
        public class SplineContainerSnapshot
        {
            public string objectPath;
            public List<SplineSnapshot> splines = new();
        }

        [Serializable]
        public class LineRendererSnapshot
        {
            public string objectPath;
            public bool useWorldSpace;
            public Vector3[] positions;
        }

        public List<SplineContainerSnapshot> splineContainers = new();
        public List<LineRendererSnapshot> lineRenderers = new();
    }

    public static class CurveDefaultsAssetCreateMenu
    {
        [MenuItem("Assets/Create/Muscles/Curve Defaults Asset (Legacy)")]
        public static void Create()
        {
            var asset = ScriptableObject.CreateInstance<CurveDefaultsAsset>();
            string path = AssetDatabase.GenerateUniqueAssetPath("Assets/CurveDefaultsAsset.asset");
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            EditorGUIUtility.PingObject(asset);
            Selection.activeObject = asset;
        }
    }
}
#endif