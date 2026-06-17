using FaceMuscle.FaceRigPipeline.Core;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Drivers
{
    public class JawDriver : MonoBehaviour, IPipelineStep
    {
        [Header("Jaw Open (Rotation)")]
        [SerializeField] private Transform            _currentPosition;
        [SerializeField] private Transform            _neutralLocalRotation;
        [SerializeField] private FaceMuscleAnchorTag  _strainTag = FaceMuscleAnchorTag.BridgeOfTheNose;
        [SerializeField] private Vector3              _rotationAxis = Vector3.right;
        [SerializeField] private float                _maxAngle = 28f;
        [SerializeField] private float                _strainForMaxAngle = 0.12f;

        [Header("Jaw Slide (Translation)")]
        [SerializeField] private MotionCaptureCalibrationData _calibrationData;
        [SerializeField] private float    _slideForMaxActivation = 0.15f;
        [SerializeField] private Transform _neutralLocalPosition;
        [SerializeField] private Vector3   _slideAxis = Vector3.right;
        [SerializeField] private float     _slideWeight = 0.01f;

        private int  _chinLandmarkIdx    = -1;
        private int  _noseRefLandmarkIdx = -1;
        private bool _initialized;

        public void Execute(FaceRigContext ctx)
        {
            if (_currentPosition == null || _neutralLocalRotation == null) return;

            if (!_initialized)
            {
                _chinLandmarkIdx    = _calibrationData.GetIndexByTag(FaceLandMarkTag.Chin);
                _noseRefLandmarkIdx = _calibrationData.GetIndexByTag(FaceLandMarkTag.BridgeOfNose);
                _initialized = true;
            }

            ApplyRotation(ctx);

            // If JawSlide already in Strains (filled by PlaybackStep) — use it.
            // Otherwise compute from landmarks and store for recording.
            float slide;
            if (ctx.Strains.TryGetValue(FaceMuscleAnchorTag.JawSlide, out slide))
            {
                // Playback or editor preview — use recorded value
            }
            else
            {
                slide = SlideJaw(ctx);
                ctx.Strains[FaceMuscleAnchorTag.JawSlide] = slide;
            }

            ConvertSlideToActivations(slide, out float jawLeft, out float jawRight);
            ApplySlide(jawLeft, jawRight);
        }

        private void ApplyRotation(FaceRigContext ctx)
        {
            if (!ctx.Strains.TryGetValue(_strainTag, out float strain)) return;

            float openStrain = Mathf.Max(0f, strain);
            float t     = Mathf.Clamp01(openStrain / _strainForMaxAngle);
            float angle = Mathf.Lerp(0f, _maxAngle, t);

            _currentPosition.localRotation = _neutralLocalRotation.localRotation.normalized
                * Quaternion.AngleAxis(angle, _rotationAxis.normalized);
        }

        private float SlideJaw(FaceRigContext ctx)
        {
            if (_chinLandmarkIdx < 0 || _noseRefLandmarkIdx < 0) return 0f;
            if (!ctx.Frame.IsValid) return 0f;

            var frame = ctx.Frame;

            float neutralVerticalDist = Mathf.Abs(frame.Neutral[_chinLandmarkIdx].y - frame.Neutral[_noseRefLandmarkIdx].y);
            if (neutralVerticalDist < 0.0001f) return 0f;

            float currentRelX = frame.Current[_chinLandmarkIdx].x - frame.Current[_noseRefLandmarkIdx].x;
            float neutralRelX = frame.Neutral[_chinLandmarkIdx].x - frame.Neutral[_noseRefLandmarkIdx].x;

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
    }
}