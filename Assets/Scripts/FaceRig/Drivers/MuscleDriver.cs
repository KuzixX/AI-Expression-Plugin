using FaceMuscle.FaceRigPipeline.Core;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Drivers
{
    public class MuscleDriver : MonoBehaviour, IPipelineStep
    {
        [SerializeField] private JointTarget[]            targets;
        [SerializeField] private Transform                currentPosition;
        [SerializeField] private Transform                neutralPosition;

        public void Execute(FaceRigContext ctx)
        {
            if (ctx.Activations == null) return;

            var activations = ctx.Activations;

            Vector3 basePos = neutralPosition.position;
            Vector3 offset  = Vector3.zero;

            foreach (var target in targets)
            {
                if (!target.target) continue;

                if (activations.TryGetValue(target.targetTag, out var activation))
                {
                    Vector3 toTarget = target.target.position - basePos;
                    offset += toTarget * (activation * target.weight);
                    target.activation = activation;
                }
            }
            currentPosition.position = basePos + offset;
        }
    }
}