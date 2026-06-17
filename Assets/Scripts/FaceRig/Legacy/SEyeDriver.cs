using FaceRig.Data;
using FaceRig.Legacy;
using UnityEngine;

namespace FaceMuscle.Runtime.Systems
{
    public class SEyeDriver : FaceDriverBase
    {
        [SerializeField] private FaceLandMarkTag eyeLandmarkTag;
        [SerializeField] private Transform       _eyeTransform;
        [SerializeField] private Transform       _eyeNeutralPosition;
        [SerializeField] private Transform       _target;
        [SerializeField] private float           _irisScale = 0.01f;

        private void Update() => Apply();

        public override void Apply()
        {
            if (ActivationStream == null) return;
            if (!ActivationStream.IrisLocalPositionStream.TryGetValue(eyeLandmarkTag, out var iris)) return;
            if (_eyeTransform == null || _eyeNeutralPosition == null || _target == null) return;

            // iris.x / iris.y — дельта от нейтрали, в нейтральной позе = (0, 0)
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