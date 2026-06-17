using System;
using FaceMuscle.FaceRigPipeline.Core;
using FaceRig.Core;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Drivers
{
    public class RotationDriver : MonoBehaviour, IPipelineStep
    {
        [SerializeField] private RotationTarget[] targets;
        [SerializeField] private Transform        currentTransform;
        [SerializeField] private Transform        neutralTransform;

        public void Execute(FaceRigContext ctx)
        {
            if (currentTransform == null || neutralTransform == null) return;

            Quaternion totalRotation = Quaternion.identity;

            foreach (var target in targets)
            {
                float activation = 0f;

                if (target.source == ActivationSource.Activation)
                {
                    if (!ctx.Activations.TryGetValue(target.tag, out activation)) continue;
                }
                else
                {
                    if (!ctx.Strains.TryGetValue(target.tag, out activation)) continue;
                }

                float angle = activation * target.maxAngle * target.weight;
                totalRotation *= Quaternion.AngleAxis(angle, target.axis.normalized);
            }

            currentTransform.localRotation = neutralTransform.localRotation * totalRotation;
        }
    }

    [Serializable]
    public class RotationTarget
    {
        public FaceMuscleAnchorTag tag;
        public ActivationSource   source = ActivationSource.Activation;
        public Vector3             axis   = Vector3.right;
        public float               maxAngle = 30f;
        public float               weight   = 1f;
    }

    public enum ActivationSource
    {
        Activation,
        Strain
    }
}
