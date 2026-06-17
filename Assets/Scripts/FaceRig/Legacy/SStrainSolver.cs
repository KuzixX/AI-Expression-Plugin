using FaceMuscle.MotionCapture;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.MotionCapture.Systems
{
    public class SStrainSolver
    {
        public float ComputeStrainBetween(int idxA, int idxB, in LandmarkFrame frame)
        {
            if (!frame.IsValid) return 0f;

            Vector3 currentA = frame.Current[idxA];
            Vector3 currentB = frame.Current[idxB];

            Vector3 neutralA = frame.Neutral[idxA];
            Vector3 neutralB = frame.Neutral[idxB];

            float restLength = Vector3.Distance(neutralA, neutralB);
            if (restLength < 0.0001f)
                return 0f;

            float currentLength = Vector3.Distance(currentA, currentB);

            return (currentLength - restLength) / restLength;
        }
    }
}
