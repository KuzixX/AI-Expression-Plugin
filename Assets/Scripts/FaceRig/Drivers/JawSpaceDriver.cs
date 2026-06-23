using FaceRig.Core;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Drivers
{
    /// <summary>
    /// Переносит 2D-движение одного лендмарка (по индексу) на джойнт, раскладывая
    /// смещение по локальным осям объекта "Jaw Space".
    ///
    ///   d  = landmark.xy(current) - landmark.xy(neutral)        // движение лендмарка
    ///   offset = (jawSpace.axisX * d.x + jawSpace.axisY * d.y) * gain
    ///   joint.position = jointNeutral.position + offset
    ///
    /// То есть "экранные" x/y лендмарка кладутся на плоскость, которую задаёт
    /// ориентация (локальные оси) объекта Jaw Space. Сам Jaw Space не двигается —
    /// он только система координат / плоскость. Двигается только позиция джойнта.
    ///
    /// Один тег = один лендмарк → один джойнт. Дублируй компонент по костям.
    /// </summary>
    public class JawSpaceDriver : MonoBehaviour, IPipelineStep
    {
        public enum LocalAxis { Right, Up, Forward }

        [Header("Landmark")]
        [Tooltip("Индекс лендмарка, с которого берём движение (в ctx.Frame).")]
        [SerializeField] private int _landmarkIndex = -1;

        [Header("Joints")]
        [Tooltip("Джойнт, который реально двигаем.")]
        [SerializeField] private Transform _currentPosition;
        [Tooltip("Нейтральная (rest) позиция джойнта — база, от которой смещаем.")]
        [SerializeField] private Transform _neutralPosition;
        [Tooltip("Объект Jaw Space: его локальные оси задают плоскость движения.")]
        [SerializeField] private Transform _jawSpace;

        [Header("Mapping landmark.xy → Jaw Space plane")]
        [Tooltip("На какую локальную ось Jaw Space ложится X лендмарка.")]
        [SerializeField] private LocalAxis _landmarkXAxis = LocalAxis.Right;
        [Tooltip("На какую локальную ось Jaw Space ложится Y лендмарка.")]
        [SerializeField] private LocalAxis _landmarkYAxis = LocalAxis.Up;
        [SerializeField] private bool _invertX;
        [SerializeField] private bool _invertY;

        [Tooltip("Масштаб: единицы лендмарка → единицы сцены.")]
        [SerializeField] private float _gain = 1f;

        // последнее посчитанное смещение — удобно для дебага в Inspector
        [SerializeField] private Vector3 _lastOffset;

        public void Execute(FaceRigContext ctx)
        {
            if (_currentPosition == null || _neutralPosition == null || _jawSpace == null) return;
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
                case LocalAxis.Right:   return _jawSpace.right;
                case LocalAxis.Up:      return _jawSpace.up;
                case LocalAxis.Forward: return _jawSpace.forward;
                default:                return Vector3.zero;
            }
        }
    }
}
