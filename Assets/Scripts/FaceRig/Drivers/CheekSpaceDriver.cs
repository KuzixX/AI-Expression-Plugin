using FaceRig.Core;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Drivers
{
    /// <summary>
    /// Драйвер щеки — та же логика, что у <see cref="JawSpaceDriver"/>:
    /// переносит 2D-движение лендмарка (по индексу) на джойнт щеки, раскладывая
    /// смещение по локальным осям объекта "Cheek Space".
    ///
    ///   d  = landmark.xy(current) - landmark.xy(neutral)        // движение лендмарка
    ///   offset = (cheekSpace.axisX * d.x + cheekSpace.axisY * d.y) * gain
    ///   joint.position = jointNeutral.position + offset
    ///
    /// Cheek Space не двигается — он только задаёт плоскость своими локальными
    /// осями. Двигается только позиция джойнта. Один компонент = один лендмарк →
    /// один джойнт; дублируй по костям (L_CHEEK / R_CHEEK).
    /// </summary>
    public class CheekSpaceDriver : MonoBehaviour, IPipelineStep
    {
        public enum LocalAxis { Right, Up, Forward }

        [Header("Landmark")]
        [Tooltip("Индекс лендмарка, с которого берём движение (в ctx.Frame).")]
        [SerializeField] private int _landmarkIndex = -1;

        [Header("Joints")]
        [Tooltip("Джойнт щеки, который реально двигаем.")]
        [SerializeField] private Transform _currentPosition;
        [Tooltip("Нейтральная (rest) позиция джойнта — база, от которой смещаем.")]
        [SerializeField] private Transform _neutralPosition;
        [Tooltip("Объект Cheek Space: его локальные оси задают плоскость движения.")]
        [SerializeField] private Transform _cheekSpace;

        [Header("Mapping landmark.xy → Cheek Space plane")]
        [Tooltip("На какую локальную ось Cheek Space ложится X лендмарка.")]
        [SerializeField] private LocalAxis _landmarkXAxis = LocalAxis.Right;
        [Tooltip("На какую локальную ось Cheek Space ложится Y лендмарка.")]
        [SerializeField] private LocalAxis _landmarkYAxis = LocalAxis.Up;
        [SerializeField] private bool _invertX;
        [SerializeField] private bool _invertY;

        [Tooltip("Масштаб: единицы лендмарка → единицы сцены.")]
        [SerializeField] private float _gain = 1f;

        // последнее посчитанное смещение — удобно для дебага в Inspector
        [SerializeField] private Vector3 _lastOffset;

        public void Execute(FaceRigContext ctx)
        {
            if (_currentPosition == null || _neutralPosition == null || _cheekSpace == null) return;
            if (!ctx.Frame.IsValid) return;

            var cur = ctx.Frame.Current;
            var neu = ctx.Frame.Neutral;
            if (_landmarkIndex < 0 || _landmarkIndex >= cur.Count || _landmarkIndex >= neu.Count) return;

            Vector3 lm  = cur[_landmarkIndex];
            Vector3 lm0 = neu[_landmarkIndex];

            float dx = (lm.x - lm0.x) * (_invertX ? -1f : 1f);
            float dy = (lm.y - lm0.y) * (_invertY ? -1f : 1f);

            Vector3 axisX = AxisVector(_landmarkXAxis);
            Vector3 axisY = AxisVector(_landmarkYAxis);

            Vector3 offset = (axisX * dx + axisY * dy) * _gain;
            _lastOffset = offset;

            _currentPosition.position = _neutralPosition.position + offset;
        }

        private Vector3 AxisVector(LocalAxis axis)
        {
            switch (axis)
            {
                case LocalAxis.Right:   return _cheekSpace.right;
                case LocalAxis.Up:      return _cheekSpace.up;
                case LocalAxis.Forward: return _cheekSpace.forward;
                default:                return Vector3.zero;
            }
        }
    }
}
