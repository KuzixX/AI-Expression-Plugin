using FaceMuscle.MotionCapture.Systems;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Strain Solver")]
    public class StrainSolverStepAsset : PipelineStepAsset
    {
        [SerializeField] private MotionCaptureCalibrationData calibrationData;
        public override IPipelineStep CreateStep() => new StrainSolverStep(calibrationData);
    }

    public class StrainSolverStep : IPipelineStep
    {
        private readonly SStrainSolver                _solver = new();
        private readonly MotionCaptureCalibrationData _calibrationData;
        public StrainSolverStep(MotionCaptureCalibrationData muscleCalibrationData)
        {
            _calibrationData = muscleCalibrationData;
        }

        public void Execute(FaceRigContext ctx)
        {
            if (!ctx.Frame.IsValid) return;

                var descriptions = _calibrationData.GetFaceLandmarkDescriptions();

                for (int i = 0; i < descriptions.Count; i++)
                {
                    var flm = descriptions[i];

                for (int j = 0; j < flm.targets.Count; j++)
                {
                    var target = flm.targets[j];
                    if (flm.idx < 0 || target.idx < 0) continue;
                    if (flm.idx >= ctx.Frame.Current.Count || target.idx >= ctx.Frame.Current.Count) continue;
                    float strain = _solver.ComputeStrainBetween(flm.idx, target.idx, in ctx.Frame);
                    ctx.Strains[target.faceMuscleAnchorTag] = strain;
                }
            }
        }
    }
}