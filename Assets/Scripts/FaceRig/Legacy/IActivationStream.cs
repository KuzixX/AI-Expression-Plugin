using System.Collections.Generic;
using FaceRig.Data;
using UnityEngine;

namespace FaceRig.Legacy
{
    public interface IActivationStream
    {
        IReadOnlyDictionary<FaceMuscleAnchorTag, float> Activations { get; }
        IReadOnlyDictionary<FaceMuscleAnchorTag, float> Strains { get; }
        IReadOnlyDictionary<FaceLandMarkTag, Vector2> IrisLocalPositionStream { get; }
        LandmarkFrame LastFrame { get; }
    }
}
