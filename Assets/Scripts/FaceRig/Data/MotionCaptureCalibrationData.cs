using System.Collections.Generic;
using UnityEngine;
namespace FaceRig.Data
{
    [CreateAssetMenu(fileName = "MotionCaptureCalibrationData", menuName = "MotionCapture/Motion Capture Calibration Data")]
    public class MotionCaptureCalibrationData : ScriptableObject
    {
        [SerializeField] private List<FaceLandmarkDescription> _faceAnchorDescription = new();
        [SerializeField] private List<EyeLandmarkDescription> _cEyeLandmarkDescriptions = new();
    
        [SerializeField] private List<Vector3> _neutralFaceLandMarks = new();

        private Dictionary<FaceLandMarkTag, int> _tagToIdx;

        public void SetLandmarks(List<Vector3> landmarks)
        {
            if(landmarks == null || landmarks.Count == 0) return;
            _neutralFaceLandMarks.Clear();
            _neutralFaceLandMarks.AddRange(landmarks);
        }

        public IReadOnlyList<Vector3> GetLandmarks()
        {
            if (_neutralFaceLandMarks == null || _neutralFaceLandMarks.Count == 0) return null;
            return _neutralFaceLandMarks;
        }

        public List<FaceLandmarkDescription> GetFaceLandmarkDescriptions()
        {
            return _faceAnchorDescription;
        }

        public List<EyeLandmarkDescription> GetEyeLandmarkDescriptions()
        {
            return _cEyeLandmarkDescriptions;
        }

        public int GetIndexByTag(FaceLandMarkTag tag)
        {
            if (_tagToIdx == null)
                BuildTagToIdxCache();

            return _tagToIdx.TryGetValue(tag, out int idx) ? idx : -1;
        }

        private void BuildTagToIdxCache()
        {
            _tagToIdx = new Dictionary<FaceLandMarkTag, int>();
            for (int i = 0; i < _faceAnchorDescription.Count; i++)
            {
                var desc = _faceAnchorDescription[i];
                _tagToIdx[desc.faceLandMarkTag] = desc.idx;
            }
        }
    }
}
