using System.Collections.Generic;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Landmark Capture")]
    public class LandmarkCaptureStepAsset : PipelineStepAsset
    {
        [SerializeField] private MotionCaptureCalibrationData calibrationData;

        public override IPipelineStep CreateStep() => new LandmarkCaptureStep(calibrationData);
    }

    public class LandmarkCaptureStep : IPipelineStep
    {
        private readonly MotionCaptureCalibrationData _calibrationData;
        private readonly List<Vector3> _currentPositions = new();
        private readonly List<Vector3> _neutralPositions = new();

        private int _upperLipCenterIdx = -1;
        private int _downLipCenterIdx  = -1;
        private bool _initialized;

        public LandmarkCaptureStep(MotionCaptureCalibrationData calibrationData)
        {
            _calibrationData = calibrationData;
        }

        public void Execute(FaceRigContext ctx)
        {
            if (!_initialized)
            {
                _upperLipCenterIdx = _calibrationData.GetIndexByTag(FaceLandMarkTag.UpperLipCenter);
                _downLipCenterIdx  = _calibrationData.GetIndexByTag(FaceLandMarkTag.DownLipCenter);
                _initialized = true;
            }

            ctx.Frame = CaptureFrame(ctx);
        }

        private LandmarkFrame CaptureFrame(FaceRigContext ctx)
        {
            if (ctx.LandmarkerRunner == null) return default;

            var result = ctx.LandmarkerRunner.GetLatestResult();
            if (result.faceLandmarks == null || result.faceLandmarks.Count == 0)
                return default;

            var landmarks = result.faceLandmarks[0].landmarks;
            var neutral = _calibrationData.GetLandmarks();
            if (neutral == null) return default;

            _currentPositions.Clear();
            _neutralPositions.Clear();

            for (int i = 0; i < landmarks.Count; i++)
            {
                var lm = landmarks[i];
                _currentPositions.Add(new Vector3(lm.x, lm.y, lm.z));
            }

            for (int i = 0; i < neutral.Count; i++)
            {
                _neutralPositions.Add(neutral[i]);
            }

            AppendMouthCenter();

            return new LandmarkFrame(_currentPositions, _neutralPositions);
        }

        private void AppendMouthCenter()
        {
            if (_upperLipCenterIdx < 0 || _downLipCenterIdx < 0) return;

            Vector3 currentCenter = (_currentPositions[_upperLipCenterIdx] + _currentPositions[_downLipCenterIdx]) * 0.5f;
            Vector3 neutralCenter = (_neutralPositions[_upperLipCenterIdx] + _neutralPositions[_downLipCenterIdx]) * 0.5f;

            _currentPositions.Add(currentCenter);
            _neutralPositions.Add(neutralCenter);
        }
    }
}
