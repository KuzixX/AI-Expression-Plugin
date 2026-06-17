using FaceMuscle.FaceRigPipeline;
using UnityEngine;

namespace FaceRig.Core
{
    // Пайплайн — ассет
    [CreateAssetMenu(menuName = "FaceRig/Pipeline")]
    public class FaceRigPipeline : ScriptableObject
    {
        public PipelineStepAsset[] steps;
    }
}