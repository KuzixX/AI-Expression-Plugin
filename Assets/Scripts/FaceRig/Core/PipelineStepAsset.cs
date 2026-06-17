using UnityEngine;

namespace FaceRig.Core
{
    public abstract class PipelineStepAsset : ScriptableObject
    {
        public abstract IPipelineStep CreateStep();
    }
}
