using System;
using UnityEngine;

namespace FaceRig.Data
{
    [Serializable]
    public class JointTarget
    {
        public FaceMuscleAnchorTag targetTag;        
        public Transform      target;
        [Range(0f,1f)]
        public float          activation;
        public float          weight = 1;
    }
}