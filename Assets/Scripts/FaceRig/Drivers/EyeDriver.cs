using FaceMuscle.FaceRigPipeline.Core;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Drivers
{
    public class EyeDriver : MonoBehaviour, IPipelineStep
    {
        [SerializeField] private FaceLandMarkTag _eyeLandmarkTag;
        [SerializeField] private Transform       _eyeTransform;
        [SerializeField] private Transform       _eyeNeutralPosition;
        [SerializeField] private Transform       _target;
        [SerializeField] private float           _irisScale = 0.01f;

        public void Execute(FaceRigContext ctx)
        {
            if (_eyeTransform == null || _eyeNeutralPosition == null || _target == null) return;
            if (!ctx.IrisPositions.TryGetValue(_eyeLandmarkTag, out var iris)) return;

            Vector3 offset = new Vector3(
                iris.x * _irisScale,
                iris.y * _irisScale,
                0f
            );

            _target.position = _eyeNeutralPosition.position + offset;
            _eyeTransform.LookAt(_target);
        }
    }
}