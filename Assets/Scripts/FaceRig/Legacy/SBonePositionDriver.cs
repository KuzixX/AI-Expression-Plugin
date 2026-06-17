using FaceRig.Data;
using FaceRig.Legacy;
using UnityEngine;

namespace FaceMuscle.Runtime.Systems
{
    public class SBonePositionDriver : FaceDriverBase
    {
        [SerializeField] private JointTarget[]            targets;
        [SerializeField] private Transform                _currentPosition;
        [SerializeField] private Transform                _neutralPosition;

        public JointTarget[]  Targets          => targets;
        public Transform      NeutralPosition  => _neutralPosition;
        public Transform      CurrentPosition  => _currentPosition;

        private void Update() { Apply(); }

        public override void Apply()
        {
            if (ActivationStream == null) return;

            var activations = ActivationStream.Activations;

            Vector3 basePos = _neutralPosition.position;
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
            _currentPosition.position = basePos + offset;
        }
    }
}
