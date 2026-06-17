using System.Collections.Generic;
using FaceRig.Data;
using Mediapipe.Unity.Sample.FaceLandmarkDetection;
using UnityEngine;

namespace FaceRig.Core
{
    public class FaceRigContext
    {
        public LandmarkFrame Frame;
        public FaceLandmarkerRunner LandmarkerRunner;
        public readonly Dictionary<FaceMuscleAnchorTag, float> Strains       = new();
        public readonly Dictionary<FaceMuscleAnchorTag, float> Activations   = new();
        public readonly Dictionary<FaceLandMarkTag, Vector2>   IrisPositions = new();

        public void Clear()
        {
            Strains.Clear();
            Activations.Clear();
            IrisPositions.Clear();
            Frame = default;
            // LandmarkerRunner НЕ чистим — это ссылка на сцену, живёт весь лайфтайм
        }
    }
}