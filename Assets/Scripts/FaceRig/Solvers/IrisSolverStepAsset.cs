using System.Collections.Generic;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Iris Solver")]
    public class IrisSolverStepAsset : PipelineStepAsset
    {
        [SerializeField] private MotionCaptureCalibrationData _calibrationData;

        public override IPipelineStep CreateStep()
        {
            var descriptions = _calibrationData.GetEyeLandmarkDescriptions();
            return new IrisSolverStep(descriptions);
        }
    }

    public class IrisSolverStep : IPipelineStep
    {
        private readonly List<EyeLandmarkDescription> _descriptions;

        public IrisSolverStep(List<EyeLandmarkDescription> descriptions)
        {
            _descriptions = descriptions;
        }

        public void Execute(FaceRigContext ctx)
        {
            if (!ctx.Frame.IsValid) return;

            var current = ctx.Frame.Current;
            var neutral = ctx.Frame.Neutral;

            for (int i = 0; i < _descriptions.Count; i++)
            {
                var desc = _descriptions[i];

                // Текущий фрейм
                Vector2 xAxisCurrent = new Vector2(
                    current[desc.eyeXAxisSpaceDescrtiption.StartIdx].x,
                    current[desc.eyeXAxisSpaceDescrtiption.EndIdx].x);
                Vector2 yAxisCurrent = new Vector2(
                    current[desc.eyeYAxisSpaceDescrtiption.StartIdx].y,
                    current[desc.eyeYAxisSpaceDescrtiption.EndIdx].y);

                float xRangeCurrent = xAxisCurrent.y - xAxisCurrent.x;
                float yRangeCurrent = yAxisCurrent.y - yAxisCurrent.x;

                if (Mathf.Abs(xRangeCurrent) < 0.0001f || Mathf.Abs(yRangeCurrent) < 0.0001f) continue;

                float currentLocalX = (current[desc.eyeIdx].x - xAxisCurrent.x) / xRangeCurrent;
                float currentLocalY = (current[desc.eyeIdx].y - yAxisCurrent.x) / yRangeCurrent;

                // Нейтральный фрейм
                Vector2 xAxisNeutral = new Vector2(
                    neutral[desc.eyeXAxisSpaceDescrtiption.StartIdx].x,
                    neutral[desc.eyeXAxisSpaceDescrtiption.EndIdx].x);
                Vector2 yAxisNeutral = new Vector2(
                    neutral[desc.eyeYAxisSpaceDescrtiption.StartIdx].y,
                    neutral[desc.eyeYAxisSpaceDescrtiption.EndIdx].y);

                float xRangeNeutral = xAxisNeutral.y - xAxisNeutral.x;
                float yRangeNeutral = yAxisNeutral.y - yAxisNeutral.x;

                if (Mathf.Abs(xRangeNeutral) < 0.0001f || Mathf.Abs(yRangeNeutral) < 0.0001f) continue;

                float neutralLocalX = (neutral[desc.eyeIdx].x - xAxisNeutral.x) / xRangeNeutral;
                float neutralLocalY = (neutral[desc.eyeIdx].y - yAxisNeutral.x) / yRangeNeutral;

                ctx.IrisPositions[desc.faceLandMarkTag] = new Vector2(
                    currentLocalX - neutralLocalX,
                    currentLocalY - neutralLocalY
                );
            }
        }
    }
}
