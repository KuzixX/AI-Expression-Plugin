using System.Collections.Generic;
using UnityEngine;

namespace FaceRig.Data
{
    public readonly struct LandmarkFrame
    {
        public readonly IReadOnlyList<Vector3> Current;
        public readonly IReadOnlyList<Vector3> Neutral;
        public readonly bool IsValid;

        public LandmarkFrame(IReadOnlyList<Vector3> current, IReadOnlyList<Vector3> neutral)
        {
            Current = current;
            Neutral = neutral;
            IsValid = current != null && neutral != null && current.Count > 0;
        }
    }
}
