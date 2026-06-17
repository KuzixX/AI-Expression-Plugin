using System.Collections.Generic;
using FaceMuscle.MotionCapture;
using FaceMuscle.MotionCapture.Systems;
using FaceMuscle.Runtime.Systems;
using FaceRig.Data;
using FaceRig.Legacy;
using Mediapipe.Unity.Sample.FaceLandmarkDetection;
using UnityEngine;

public class SStreamMCActivationRAW : MonoBehaviour, IActivationStream
{
    [SerializeField] private FaceLandmarkerRunner              faceLandmarkerResult;
    [SerializeField] private MotionCaptureCalibrationData      motionCaptureCalibrationData;
    [SerializeField] private List<FaceLandmarkDescription>    faceLandmarkDescription = new();
    [SerializeField] private List<EyeLandmarkDescription>     cEyeLandmarkDescription = new();
    private readonly Dictionary<FaceMuscleAnchorTag, float>    _activationsStream = new();
    private readonly Dictionary<FaceMuscleAnchorTag, float>    _strainStream = new();
    private readonly Dictionary<FaceLandMarkTag, Vector2>      _irisLocalPositionStream = new();
    
    private readonly SStrainSolver     _strainSolver     = new();
    private readonly SActivationSolver _activationSolver = new();

    private readonly List<Vector3>     _currentFramePositions = new();
    private readonly List<Vector3>     _neutralFramePositions = new();

    private int _upperLipCenterIdx = -1;
    private int _downLipCenterIdx  = -1;

    public int MouthCenterLandmarkIndex { get; private set; } = -1;

    public IReadOnlyDictionary<FaceMuscleAnchorTag, float>   Activations              => _activationsStream;
    public IReadOnlyDictionary<FaceMuscleAnchorTag, float>   Strains                  => _strainStream;
    
    public IReadOnlyDictionary<FaceLandMarkTag, Vector2>     IrisLocalPositionStream  => _irisLocalPositionStream;
    public LandmarkFrame LastFrame { get; private set; }

    private void Start()
    {
        BuildRuntimeFaceLandmarkDescriptions();
        BuildRuntimeEyeLandmarkDescription();
        
        _upperLipCenterIdx = motionCaptureCalibrationData.GetIndexByTag(FaceLandMarkTag.UpperLipCenter);
        _downLipCenterIdx  = motionCaptureCalibrationData.GetIndexByTag(FaceLandMarkTag.DownLipCenter);
    }

    private void Update()
    {
        UpdateActivations();
        UpdateIrisLocalPositions();
    }

    private void BuildRuntimeEyeLandmarkDescription()
    {
        var source = motionCaptureCalibrationData.GetEyeLandmarkDescriptions();
        cEyeLandmarkDescription = new List<EyeLandmarkDescription>();

        for (int i = 0; i < source.Count; i++)
        {
            var src = source[i];
            EyeLandmarkDescription clone = new EyeLandmarkDescription();
            clone.faceLandMarkTag = src.faceLandMarkTag;
            clone.eyeIdx          = src.eyeIdx;
            clone.eyeXAxisSpaceDescrtiption   = src.eyeXAxisSpaceDescrtiption;
            clone.eyeYAxisSpaceDescrtiption   = src.eyeYAxisSpaceDescrtiption;
            cEyeLandmarkDescription.Add(clone);
        }
    }
    private void BuildRuntimeFaceLandmarkDescriptions()
    {
        var source = motionCaptureCalibrationData.GetFaceLandmarkDescriptions();
        faceLandmarkDescription = new List<FaceLandmarkDescription>();

        for (int i = 0; i < source.Count; i++)
        {
            var src = source[i];

            FaceLandmarkDescription clone = new FaceLandmarkDescription();
            clone.faceLandMarkTag = src.faceLandMarkTag;
            clone.idx = src.idx;
            clone.targets = new List<FaceLandmarkTargetDescription>();

            for (int j = 0; j < src.targets.Count; j++)
            {
                var srcTarget = src.targets[j];

                FaceLandmarkTargetDescription targetClone = new FaceLandmarkTargetDescription();
                targetClone.idx = srcTarget.idx;
                targetClone.faceMuscleAnchorTag = srcTarget.faceMuscleAnchorTag;
                targetClone.activation = 0f;

                clone.targets.Add(targetClone);
            }
            faceLandmarkDescription.Add(clone);
        }
    }

    private void UpdateIrisLocalPositions()
    {
        if (!LastFrame.IsValid) return;

        var current = LastFrame.Current;
        var neutral = LastFrame.Neutral;

        for (int i = 0; i < cEyeLandmarkDescription.Count; i++)
        {
            var desc = cEyeLandmarkDescription[i];

            // Текущий фрейм
            Vector2 xAxisCurrent = new Vector2(current[desc.eyeXAxisSpaceDescrtiption.StartIdx].x, current[desc.eyeXAxisSpaceDescrtiption.EndIdx].x);
            Vector2 yAxisCurrent = new Vector2(current[desc.eyeYAxisSpaceDescrtiption.StartIdx].y, current[desc.eyeYAxisSpaceDescrtiption.EndIdx].y);

            float xRangeCurrent = xAxisCurrent.y - xAxisCurrent.x;
            float yRangeCurrent = yAxisCurrent.y - yAxisCurrent.x;

            if (Mathf.Abs(xRangeCurrent) < 0.0001f || Mathf.Abs(yRangeCurrent) < 0.0001f) continue;

            float currentLocalX = (current[desc.eyeIdx].x - xAxisCurrent.x) / xRangeCurrent;
            float currentLocalY = (current[desc.eyeIdx].y - yAxisCurrent.x) / yRangeCurrent;

            // Нейтральный фрейм
            Vector2 xAxisNeutral = new Vector2(neutral[desc.eyeXAxisSpaceDescrtiption.StartIdx].x, neutral[desc.eyeXAxisSpaceDescrtiption.EndIdx].x);
            Vector2 yAxisNeutral = new Vector2(neutral[desc.eyeYAxisSpaceDescrtiption.StartIdx].y, neutral[desc.eyeYAxisSpaceDescrtiption.EndIdx].y);

            float xRangeNeutral = xAxisNeutral.y - xAxisNeutral.x;
            float yRangeNeutral = yAxisNeutral.y - yAxisNeutral.x;

            if (Mathf.Abs(xRangeNeutral) < 0.0001f || Mathf.Abs(yRangeNeutral) < 0.0001f) continue;

            float neutralLocalX = (neutral[desc.eyeIdx].x - xAxisNeutral.x) / xRangeNeutral;
            float neutralLocalY = (neutral[desc.eyeIdx].y - yAxisNeutral.x) / yRangeNeutral;

            // Дельта: в нейтральной позе = (0, 0)
            _irisLocalPositionStream[desc.faceLandMarkTag] = new Vector2(
                currentLocalX - neutralLocalX,
                currentLocalY - neutralLocalY
            );
        }
    }

    private void UpdateActivations()
    {
        var frame = CaptureFrame();
        if (!frame.IsValid) return;
        LastFrame = frame;

        for (int i = 0; i < faceLandmarkDescription.Count; i++)
        {
            var flm = faceLandmarkDescription[i];

            for (int j = 0; j < flm.targets.Count; j++)
            {
                var flmT = flm.targets[j];

                float strain     = _strainSolver.ComputeStrainBetween(flm.idx, flmT.idx, in frame);
                float activation = _activationSolver.ComputeActivationFromStrain(strain);

                flmT.activation  = activation;
                _activationsStream[flmT.faceMuscleAnchorTag] = activation;
                _strainStream[flmT.faceMuscleAnchorTag]      = strain;
            }
        }
    }

    private LandmarkFrame CaptureFrame()
    {
        var result = faceLandmarkerResult.GetLatestResult();
        if (result.faceLandmarks == null || result.faceLandmarks.Count == 0)
            return default;

        var landmarks = result.faceLandmarks[0].landmarks;
        var neutral = motionCaptureCalibrationData.GetLandmarks();
        if (neutral == null) return default;

        _currentFramePositions.Clear();
        _neutralFramePositions.Clear();

        for (int i = 0; i < landmarks.Count; i++)
        {
            var lm = landmarks[i];
            _currentFramePositions.Add(new Vector3(lm.x, lm.y, lm.z));
        }

        for (int i = 0; i < neutral.Count; i++)
        {
            _neutralFramePositions.Add(neutral[i]);
        }

        AppendVirtualLandmarks();

        return new LandmarkFrame(_currentFramePositions, _neutralFramePositions);
    }

    private void AppendVirtualLandmarks()
    {
        AppendMouthCenter();
    }

    private void AppendMouthCenter()
    {
        if (_upperLipCenterIdx < 0 || _downLipCenterIdx < 0) return;

        Vector3 currentCenter = (_currentFramePositions[_upperLipCenterIdx] + _currentFramePositions[_downLipCenterIdx]) * 0.5f;
        Vector3 neutralCenter = (_neutralFramePositions[_upperLipCenterIdx] + _neutralFramePositions[_downLipCenterIdx]) * 0.5f;

        MouthCenterLandmarkIndex = _currentFramePositions.Count;
        _currentFramePositions.Add(currentCenter);
        _neutralFramePositions.Add(neutralCenter);
    }

    public List<FaceLandmarkDescription> GetRuntimeFaceLandmarkDescription()
    {
        return faceLandmarkDescription;
    }
}
