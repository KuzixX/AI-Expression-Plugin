using System;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "BoneDefaultPose", menuName = "Rig/Bone Default Pose")]
public class BoneDefaultPose : ScriptableObject
{
    [Serializable]
    public class BoneData
    {
        public string path;              // Relative path from root
        public Vector3 localPosition;
        public Quaternion localRotation;
        public Vector3 localScale;
    }

    public List<BoneData> bones = new List<BoneData>();
}