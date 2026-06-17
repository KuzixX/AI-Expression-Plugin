using UnityEngine;

namespace FaceMuscle.MotionCapture.Systems
{
    public class SDebugLandmarks : MonoBehaviour
    {
        [SerializeField] private RectTransform canvas;

        private void DrawDebugLineOnCanvas(Vector3 fromNormalized, Vector3 toNormalized, Color color)
        {
            Vector2 fromLocal = NormalizedToCanvasLocal(fromNormalized);
            Vector2 toLocal   = NormalizedToCanvasLocal(toNormalized);

            Vector3 fromWorld = canvas.TransformPoint(fromLocal);
            Vector3 toWorld   = canvas.TransformPoint(toLocal);

            Debug.DrawLine(fromWorld, toWorld, color);
        }

        private Vector2 NormalizedToCanvasLocal(Vector3 normalizedPos)
        {
            return new Vector2(
                (normalizedPos.x - 0.5f) * canvas.rect.width,
                (0.5f - normalizedPos.y) * canvas.rect.height
            );
        }
    }
}