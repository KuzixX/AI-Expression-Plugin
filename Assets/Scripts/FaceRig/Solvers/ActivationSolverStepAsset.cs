using FaceRig.Core;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Activation Solver")]
    public class ActivationSolverStepAsset : PipelineStepAsset
    {
        public override IPipelineStep CreateStep() => new ActivationSolverStep();
    }

    public class ActivationSolverStep : IPipelineStep
    {
        private readonly SActivationSolver _solver = new();

        public void Execute(FaceRigContext ctx)
        {
            if (ctx.Strains.Count == 0) return;

            foreach (var kvp in ctx.Strains)
            {
                float activation = _solver.ComputeActivationFromStrain(kvp.Value);
                ctx.Activations[kvp.Key] = activation;
            }
        }
    }
}
