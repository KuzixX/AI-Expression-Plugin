using System;
using UnityEngine.Serialization;

namespace FaceRig.Data
{
    [Serializable]
    public class FaceLandmarkTargetDescription
    {
        [FormerlySerializedAs("FaceLandMarkTargetTag")]
        [FormerlySerializedAs("faceMusculeAnchorTag")]
        public FaceMuscleAnchorTag faceMuscleAnchorTag;
        public int idx;
        public float activation;
    }
}