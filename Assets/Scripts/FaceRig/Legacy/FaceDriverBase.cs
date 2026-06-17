using FaceMuscle.Runtime.Systems;
using UnityEngine;

namespace FaceRig.Legacy
{
    public abstract class FaceDriverBase : MonoBehaviour, IFaceDriver
    {
        protected IActivationStream ActivationStream { get; private set; }

        protected virtual void Start()
        {
            foreach (var mb in FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
            {
                if (mb is IActivationStream stream)
                {
                    ActivationStream = stream;
                    return;
                }
            }
        }

        public abstract void Apply();
    }
}
