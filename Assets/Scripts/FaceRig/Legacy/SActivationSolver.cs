using UnityEngine;

public class SActivationSolver
    {
        public float ComputeActivationFromStrain(float strain, float gain = 1f)
        {
            float activation = -strain * gain;
            return Mathf.Clamp01(activation);
        }
}