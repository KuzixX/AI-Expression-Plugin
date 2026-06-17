using FaceRig.Data;
using FaceRig.Legacy;
using UnityEngine;

namespace FaceMuscle.Runtime.Systems
{
    public class SJawDriver : FaceDriverBase
    {
        [SerializeField] private Transform _currentPosition;
        [SerializeField] private Transform _neutralLocalRotation;
        [SerializeField] private FaceMuscleAnchorTag _strainTag = FaceMuscleAnchorTag.BridgeOfTheNose;
        [SerializeField] private Vector3 _rotationAxis = Vector3.right;
        [SerializeField] private float _maxAngle = 28f;
        [SerializeField] private float _strainForMaxAngle = 0.12f;

        [Header("Jaw Slide")]
        [SerializeField] private MotionCaptureCalibrationData _calibrationData;
        [SerializeField] private float _slideForMaxActivation = 0.15f;
        [SerializeField] private Transform _neutralLocalPosition;
        [SerializeField] private Vector3 _slideAxis = Vector3.right;
        [SerializeField] private float _slideWeight = 0.01f;

        private int _chinLandmarkIdx    = -1;
        private int _noseRefLandmarkIdx = -1;

        protected override void Start()
        {
            base.Start();
            _chinLandmarkIdx    = _calibrationData.GetIndexByTag(FaceLandMarkTag.Chin);
            _noseRefLandmarkIdx = _calibrationData.GetIndexByTag(FaceLandMarkTag.BridgeOfNose);
        }

        private void Update()
        {
            Apply();
            ConvertSlideToActivations(SlideJaw(), out float jawLeft, out float jawRight);
            ApplySlide(jawLeft, jawRight);
        }

        public override void Apply()
        {
            if (ActivationStream == null) return;

            if (!ActivationStream.Strains.TryGetValue(_strainTag, out float strain))
                return;

            float angle = ConvertStrainToAngle(strain);

            _currentPosition.localRotation = _neutralLocalRotation.localRotation.normalized
                * Quaternion.AngleAxis(angle, _rotationAxis.normalized);
        }

        private float SlideJaw()
        {
            if (_chinLandmarkIdx < 0 || _noseRefLandmarkIdx < 0) return 0f;

            var frame = ActivationStream.LastFrame;
            if (!frame.IsValid) return 0f;

            float neutralVerticalDist = Mathf.Abs(frame.Neutral[_chinLandmarkIdx].y - frame.Neutral[_noseRefLandmarkIdx].y);
            if (neutralVerticalDist < 0.0001f) return 0f;

            float currentRelX = frame.Current[_chinLandmarkIdx].x - frame.Current[_noseRefLandmarkIdx].x;
            float neutralRelX = frame.Neutral[_chinLandmarkIdx].x  - frame.Neutral[_noseRefLandmarkIdx].x;

            return (currentRelX - neutralRelX) / neutralVerticalDist;
        }

        private void ApplySlide(float jawLeft, float jawRight)
        {
            if (_neutralLocalPosition == null) return;

            float displacement = (jawRight - jawLeft) * _slideWeight;
            _currentPosition.localPosition = _neutralLocalPosition.localPosition + _slideAxis.normalized * displacement;
        }

        private void ConvertSlideToActivations(float slide, out float jawLeft, out float jawRight)
        {
            jawRight = Mathf.Clamp01( slide / _slideForMaxActivation);
            jawLeft  = Mathf.Clamp01(-slide / _slideForMaxActivation);
        }

        private float ConvertStrainToAngle(float strain)
        {
            float openStrain = Mathf.Max(0f, strain);
            float t = Mathf.Clamp01(openStrain / _strainForMaxAngle);
            return Mathf.Lerp(0f, _maxAngle, t);
        }
    }
}
