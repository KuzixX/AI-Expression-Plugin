using System.Collections.Generic;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Temporal Smoothing")]
    public class SmoothingStepAsset : PipelineStepAsset
    {
        [Header("Exponential Moving Average")]
        [SerializeField] [Range(0f, 0.99f)] private float _smoothing = 0.5f;

        [Header("Iris")]
        [SerializeField] [Range(0f, 0.99f)] private float _irisSmooothing = 0.6f;
        [SerializeField] private float _irisClamp = 1f;

        public override IPipelineStep CreateStep()
            => new SmoothingStep(_smoothing, _irisSmooothing, _irisClamp);
    }

    public class SmoothingStep : IPipelineStep
    {
        private readonly float _smoothing;
        private readonly float _irisSmoothing;
        private readonly float _irisClamp;

        private readonly Dictionary<FaceMuscleAnchorTag, float> _prevActivations = new();
        private readonly Dictionary<FaceMuscleAnchorTag, float> _prevStrains     = new();
        private readonly Dictionary<FaceLandMarkTag, Vector2>   _prevIris        = new();

        public SmoothingStep(float smoothing, float irisSmoothing, float irisClamp)
        {
            _smoothing     = smoothing;
            _irisSmoothing = irisSmoothing;
            _irisClamp     = irisClamp;
        }

        public void Execute(FaceRigContext ctx)
        {
            // Activations
            foreach (var tag in new List<FaceMuscleAnchorTag>(ctx.Activations.Keys))
            {
                float cur = ctx.Activations[tag];

                if (_prevActivations.TryGetValue(tag, out float prev))
                    cur = Mathf.Lerp(cur, prev, _smoothing);

                _prevActivations[tag] = cur;
                ctx.Activations[tag] = cur;
            }

            // Strains
            foreach (var tag in new List<FaceMuscleAnchorTag>(ctx.Strains.Keys))
            {
                float cur = ctx.Strains[tag];

                if (_prevStrains.TryGetValue(tag, out float prev))
                    cur = Mathf.Lerp(cur, prev, _smoothing);

                _prevStrains[tag] = cur;
                ctx.Strains[tag] = cur;
            }

            // Iris: clamp + smooth
            foreach (var tag in new List<FaceLandMarkTag>(ctx.IrisPositions.Keys))
            {
                var cur = ctx.IrisPositions[tag];

                // Clamp outliers
                cur.x = Mathf.Clamp(cur.x, -_irisClamp, _irisClamp);
                cur.y = Mathf.Clamp(cur.y, -_irisClamp, _irisClamp);

                if (_prevIris.TryGetValue(tag, out var prev))
                    cur = Vector2.Lerp(cur, prev, _irisSmoothing);

                _prevIris[tag] = cur;
                ctx.IrisPositions[tag] = cur;
            }
        }
    }
}
