using System.Collections.Generic;
using FaceRig.Data;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using Mediapipe.Unity.Sample.FaceLandmarkDetection;
using UnityEngine;

public class SFaceCalibration : MonoBehaviour
{
    [SerializeField] private MotionCaptureCalibrationData motionCaptureCalibrationData;
    [SerializeField] private FaceLandmarkerRunner         faceLandmarkerRunner;

    [SerializeField] private int iterations = 6;
    [SerializeField] private float interval = 1f;

    private readonly List<Vector3> _sumPos = new();
    private int _currentIteration = 0;
    private float _timer          = 0f;
    private bool _isCalibrating   = false;

    private void Update()
    {
        if (!_isCalibrating)
            return;

        var result = faceLandmarkerRunner.GetLatestResult();

        if (result.faceLandmarks == null || result.faceLandmarks.Count == 0)
            return;

        var avg = CalibrateAvgLandmarksPos(result, iterations, interval);

        if (avg != null)
        {
            motionCaptureCalibrationData.SetLandmarks(avg);
            Debug.Log("Neutral landmarks saved");
        }
    }

    [ContextMenu("Start Calibration")]
    public void StartCalibration()
    {
        _isCalibrating = true;
        _currentIteration = 0;
        _timer = 0f;
        _sumPos.Clear();

        Debug.Log("Calibration started");
    }

    private List<Vector3> CalibrateAvgLandmarksPos(FaceLandmarkerResult result, int iterations, float interval)
    {
        _timer += Time.deltaTime;

        if (_timer < interval) return null;

        _timer = 0f;

        var landmarks = result.faceLandmarks[0].landmarks;

        if (_currentIteration == 0)
        {
            _sumPos.Clear();

            for (int i = 0; i < landmarks.Count; i++)
            {
                _sumPos.Add(Vector3.zero);
            }
        }

        for (int i = 0; i < landmarks.Count; i++)
        {
            var fl = landmarks[i];
            _sumPos[i] += new Vector3(fl.x, fl.y, fl.z);
        }

        _currentIteration++;
        Debug.Log($"Calibration iteration: {_currentIteration}/{iterations}");

        if (_currentIteration >= iterations)
        {
            var avg = new List<Vector3>(_sumPos.Count);

            for (int i = 0; i < _sumPos.Count; i++)
            {
                avg.Add(_sumPos[i] / iterations);
            }

            _currentIteration = 0;
            _isCalibrating = false;

            Debug.Log("Calibration finished");

            return avg;
        }

        return null;
    }
}