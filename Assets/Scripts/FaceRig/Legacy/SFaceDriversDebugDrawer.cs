using FaceMuscle.Runtime.Systems;
using UnityEngine;

/*public class SFaceDriversDebugDrawer : MonoBehaviour
{
    [SerializeField] private Color _neutralToTargetColor = Color.cyan;
    [SerializeField] private Color _currentToTargetColor = Color.yellow;
    [SerializeField] private bool _drawCurrentToTarget = false;
    [SerializeField] private bool _refreshDriversEveryFrame = false;

    private SBonePositionDriver[] _drivers;

    private void Awake()
    {
        CacheDrivers();
    }

   // private void Update({ if (_refreshDriversEveryFrame || _drivers == null) CacheDrivers();

     //   DrawDriverTargets();
    //}

    [ContextMenu("Cache Drivers")]
    private void CacheDrivers()
    {
#if UNITY_2023_1_OR_NEWER
        _drivers = FindObjectsByType<SBonePositionDriver>(FindObjectsSortMode.None);
#else
        _drivers = FindObjectsOfType<SBonePositionDriver>();
#endif
    }

    private void DrawDriverTargets()
    {
        if (_drivers == null) return;

        for (int i = 0; i < _drivers.Length; i++)
        {
            var driver = _drivers[i];
            if (driver == null) continue;

            var neutral = driver.NeutralPosition;
            var current = driver.CurrentPosition;
            var targets = driver.Targets;

            if (neutral == null || targets == null)
                continue;

            for (int j = 0; j < targets.Length; j++)
            {
                var jointTarget = targets[j];
                if (jointTarget == null || jointTarget.target == null)
                    continue;

                Debug.DrawLine(neutral.position, jointTarget.target.position, _neutralToTargetColor);

                if (_drawCurrentToTarget && current != null)
                    Debug.DrawLine(current.position, jointTarget.target.position, _currentToTargetColor);
            }
        }
    }
} */
