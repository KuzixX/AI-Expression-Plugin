using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace FaceRig.Data
{
    [Serializable]
    public class EyeLandmarkDescription
    {

        [SerializeField] public FaceLandMarkTag faceLandMarkTag;
        [SerializeField] public int              eyeIdx;
        [FormerlySerializedAs("eyeXAxisSpace")]
        [Header("X Axis")]
        [SerializeField] public EyeAxisSpaceDescrtiption     eyeXAxisSpaceDescrtiption;
        [FormerlySerializedAs("eyeYAxisSpace")]
        [Header("Y Axis")]
        [SerializeField] public EyeAxisSpaceDescrtiption     eyeYAxisSpaceDescrtiption;
    }
}