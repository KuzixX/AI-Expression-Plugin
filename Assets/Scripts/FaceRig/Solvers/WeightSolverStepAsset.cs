using System;
using System.Collections.Generic;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Weight Solver")]
    public class WeightSolverStepAsset : PipelineStepAsset
    {
        [SerializeField] private List<MuscleWeightEntry> muscleWeights = new();

        public override IPipelineStep CreateStep() => new WeightSolverStep(muscleWeights);
    }

    [Serializable]
    public class MuscleWeightEntry
    {
        public FaceMuscleAnchorTag tag;
        public AnimationCurve     curve = AnimationCurve.Linear(0f, 0f, 1f, 1f);
    }

    public class WeightSolverStep : IPipelineStep
    {
        private readonly Dictionary<FaceMuscleAnchorTag, AnimationCurve> _curveMap = new();

        public WeightSolverStep(List<MuscleWeightEntry> entries)
        {
            foreach (var entry in entries)
                _curveMap[entry.tag] = entry.curve;
        }

        public void Execute(FaceRigContext ctx)
        {
            if (ctx.Activations.Count == 0) return;

            foreach (var tag in new List<FaceMuscleAnchorTag>(ctx.Activations.Keys))
            {
                if (_curveMap.TryGetValue(tag, out var curve))
                    ctx.Activations[tag] = curve.Evaluate(ctx.Activations[tag]);
            }
        }
    }
}
