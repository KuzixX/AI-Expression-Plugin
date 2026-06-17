using System.Collections.Generic;
using FaceRig.Data;
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using Mediapipe.Unity.Sample.FaceLandmarkDetection;
using UnityEngine;
using UnityEngine.UI;

public class SFaceLandmarkDebugDrawer : MonoBehaviour
{
    [Header("All Landmark Indices")]
    [SerializeField] private bool _drawAllLandmarkIndices = false;
    [SerializeField] private int _allLandmarkIndexFontSize = 12;
    [SerializeField] private Vector2 _allLandmarkIndexOffset = new Vector2(6f, -6f);
    [SerializeField] private Color _allLandmarkIndexColor = Color.white;
    
    [Header("Sources")]
    [SerializeField] private FaceLandmarkerRunner _faceLandmarkerRunner;
    [SerializeField] private MotionCaptureCalibrationData _calibrationData;
    [SerializeField] private RectTransform _debugCanvas;

    [Header("Filtering")]
    [SerializeField] private bool _drawAll = true;
    [SerializeField] private List<FaceLandMarkTag> _tagsToDraw = new();

    [Header("Draw Options")]
    [SerializeField] private bool _drawLines = true;
    [SerializeField] private bool _drawPoints = true;
    [SerializeField] private bool _drawLabels = true;
    [SerializeField] private bool _drawPointIndices = true;

    [Header("Visual Settings")]
    [SerializeField] private float _pointSize = 10f;
    [SerializeField] private Font _labelFont;
    [SerializeField] private int _labelFontSize = 16;
    [SerializeField] private Vector2 _labelOffset = new Vector2(12f, 12f);

    [Header("Point Index Settings")]
    [SerializeField] private int _pointIndexFontSize = 14;
    [SerializeField] private Vector2 _pointIndexOffset = new Vector2(0f, -14f);
    [SerializeField] private Color _pointIndexColor = Color.cyan;

    [Header("Mouth Center")]
    [SerializeField] private bool _drawMouthCenter = true;
    [SerializeField] private Color _mouthCenterColor = Color.red;
    [SerializeField] private float _mouthCenterPointSize = 14f;

    [Header("Colors")]
    [SerializeField] private Color _defaultColor = Color.green;
    [SerializeField] private Color _sourcePointColor = Color.white;
    [SerializeField] private Color _targetPointColor = Color.yellow;
    [SerializeField] private Color _labelColor = Color.white;

    private int _upperLipCenterIdx = -1;
    private int _downLipCenterIdx  = -1;
    private int _lLipAngleIdx      = -1;
    private int _rLipAngleIdx      = -1;

    private readonly List<GameObject> _spawnedUi = new();

    private void Start()
    {
        CacheLipCenterIndices();
    }

    private void LateUpdate()
    {
        ClearDrawnUi();
        DrawConfiguredLandmarks();
        DrawAllLandmarkIndices();
        DrawMouthCenter();
    }
    private void DrawAllLandmarkIndices()
    {
        if (!_drawAllLandmarkIndices)
            return;

        if (_faceLandmarkerRunner == null || _debugCanvas == null)
            return;

        FaceLandmarkerResult result = _faceLandmarkerRunner.GetLatestResult();
        if (result.faceLandmarks == null || result.faceLandmarks.Count == 0)
            return;

        var landmarks = result.faceLandmarks[0].landmarks;

        for (int i = 0; i < landmarks.Count; i++)
        {
            Vector3 normalized = ToVector3(landmarks[i]);
            Vector2 canvasPos = NormalizedToCanvasLocal(normalized);

            CreateLabel(
                $"AllIndex_{i}",
                i.ToString(),
                canvasPos + _allLandmarkIndexOffset,
                _allLandmarkIndexColor,
                _allLandmarkIndexFontSize
            );
        }
    }
    private void DrawConfiguredLandmarks()
    {
        if (_faceLandmarkerRunner == null || _calibrationData == null || _debugCanvas == null)
            return;

        FaceLandmarkerResult result = _faceLandmarkerRunner.GetLatestResult();
        if (result.faceLandmarks == null || result.faceLandmarks.Count == 0)
            return;

        var landmarks = result.faceLandmarks[0].landmarks;
        var descriptions = _calibrationData.GetFaceLandmarkDescriptions();

        if (descriptions == null)
            return;

        for (int i = 0; i < descriptions.Count; i++)
        {
            var desc = descriptions[i];

            if (!ShouldDraw(desc.faceLandMarkTag))
                continue;

            if (!IsValidIndex(desc.idx, landmarks.Count))
                continue;

            Vector3 sourceNormalized = ToVector3(landmarks[desc.idx]);
            Vector2 sourceCanvasPos = NormalizedToCanvasLocal(sourceNormalized);

            if (_drawPoints)
            {
                CreatePoint(
                    $"S_{desc.idx}",
                    sourceCanvasPos,
                    _pointSize,
                    _sourcePointColor
                );
            }

            if (_drawLabels)
            {
                CreateLabel(
                    $"SLabel_{desc.idx}",
                    $"{desc.faceLandMarkTag}",
                    sourceCanvasPos + _labelOffset,
                    _labelColor,
                    _labelFontSize
                );
            }

            if (_drawPointIndices)
            {
                CreatePointIndexLabel(
                    $"SIndex_{desc.idx}",
                    desc.idx.ToString(),
                    sourceCanvasPos + _pointIndexOffset,
                    _pointIndexColor
                );
            }

            if (desc.targets == null)
                continue;

            for (int j = 0; j < desc.targets.Count; j++)
            {
                var target = desc.targets[j];

                if (!IsValidIndex(target.idx, landmarks.Count))
                    continue;

                Vector3 targetNormalized = ToVector3(landmarks[target.idx]);
                Vector2 targetCanvasPos = NormalizedToCanvasLocal(targetNormalized);

                Color linkColor = GetColorByTargetTag(target.faceMuscleAnchorTag);

                if (_drawLines)
                {
                    DrawUiLine(
                        $"Line_{desc.idx}_{target.idx}",
                        sourceCanvasPos,
                        targetCanvasPos,
                        2f,
                        linkColor
                    );
                }

                if (_drawPoints)
                {
                    CreatePoint(
                        $"T_{target.idx}",
                        targetCanvasPos,
                        _pointSize,
                        _targetPointColor
                    );
                }

                if (_drawLabels)
                {
                    CreateLabel(
                        $"TLabel_{target.idx}",
                        $"{target.faceMuscleAnchorTag}",
                        targetCanvasPos + _labelOffset,
                        linkColor,
                        _labelFontSize
                    );
                }

                if (_drawPointIndices)
                {
                    CreatePointIndexLabel(
                        $"TIndex_{target.idx}",
                        target.idx.ToString(),
                        targetCanvasPos + _pointIndexOffset,
                        _pointIndexColor
                    );
                }
            }
        }
    }

    private bool ShouldDraw(FaceLandMarkTag tag)
    {
        if (_drawAll)
            return true;

        for (int i = 0; i < _tagsToDraw.Count; i++)
        {
            if (_tagsToDraw[i] == tag)
                return true;
        }

        return false;
    }

    private bool IsValidIndex(int idx, int count)
    {
        return idx >= 0 && idx < count;
    }

    private Vector3 ToVector3(NormalizedLandmark lm)
    {
        return new Vector3(lm.x, lm.y, lm.z);
    }

    private Vector2 NormalizedToCanvasLocal(Vector3 normalizedPos)
    {
        return new Vector2(
            (normalizedPos.x - 0.5f) * _debugCanvas.rect.width,
            (0.5f - normalizedPos.y) * _debugCanvas.rect.height
        );
    }

    private void CreatePoint(string objectName, Vector2 anchoredPos, float size, Color color)
    {
        GameObject go = new GameObject(objectName, typeof(RectTransform), typeof(Image));
        go.transform.SetParent(_debugCanvas, false);

        var rect = go.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(size, size);
        rect.anchoredPosition = anchoredPos;
        rect.localScale = Vector3.one;

        var image = go.GetComponent<Image>();
        image.color = color;

        _spawnedUi.Add(go);
    }

    private void CreateLabel(string objectName, string text, Vector2 anchoredPos, Color color, int fontSize)
    {
        GameObject go = new GameObject(objectName, typeof(RectTransform), typeof(Text));
        go.transform.SetParent(_debugCanvas, false);

        var rect = go.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(260f, 30f);
        rect.anchoredPosition = anchoredPos;
        rect.localScale = Vector3.one;

        var label = go.GetComponent<Text>();
        label.text = text;
        label.color = color;
        label.fontSize = fontSize;
        label.font = _labelFont != null ? _labelFont : Resources.GetBuiltinResource<Font>("Arial.ttf");
        label.alignment = TextAnchor.MiddleLeft;
        label.horizontalOverflow = HorizontalWrapMode.Overflow;
        label.verticalOverflow = VerticalWrapMode.Overflow;

        _spawnedUi.Add(go);
    }

    private void CreatePointIndexLabel(string objectName, string text, Vector2 anchoredPos, Color color)
    {
        CreateLabel(objectName, text, anchoredPos, color, _pointIndexFontSize);
    }

    private void DrawUiLine(string objectName, Vector2 from, Vector2 to, float thickness, Color color)
    {
        GameObject go = new GameObject(objectName, typeof(RectTransform), typeof(Image));
        go.transform.SetParent(_debugCanvas, false);

        var rect = go.GetComponent<RectTransform>();
        Vector2 dir = to - from;
        float length = dir.magnitude;

        rect.sizeDelta = new Vector2(length, thickness);
        rect.anchoredPosition = from + dir * 0.5f;
        rect.localRotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg);
        rect.localScale = Vector3.one;

        var image = go.GetComponent<Image>();
        image.color = color;

        _spawnedUi.Add(go);
    }

    private Color GetColorByTargetTag(FaceMuscleAnchorTag tag)
    {
        return tag switch
        {
            FaceMuscleAnchorTag.lZygomaticusMajorLandmark => Color.yellow,
            FaceMuscleAnchorTag.rZygomaticusMajorLandmark => Color.yellow,

            FaceMuscleAnchorTag.lZygomaticusMinorLandmark => new Color(1f, 0.7f, 0.2f),
            FaceMuscleAnchorTag.rZygomaticusMinorLandmark => new Color(1f, 0.7f, 0.2f),

            FaceMuscleAnchorTag.lDepressorAnguliOrisLandmark => Color.cyan,
            FaceMuscleAnchorTag.rDepressorAnguliOrisLandmark => Color.cyan,

            FaceMuscleAnchorTag.lRisoriusLandmark => Color.magenta,
            FaceMuscleAnchorTag.rRisoriusLandmark => Color.magenta,

            FaceMuscleAnchorTag.lFrontalisInsideLandmark => Color.green,
            FaceMuscleAnchorTag.rFrontalisInsideLandmark => Color.green,

            FaceMuscleAnchorTag.lFrontalisOuterLandmark => new Color(0.3f, 1f, 0.3f),
            FaceMuscleAnchorTag.rFrontalisOuterLandmark => new Color(0.3f, 1f, 0.3f),

            FaceMuscleAnchorTag.LCorrugatorSuperciliiLandmark => Color.red,
            FaceMuscleAnchorTag.RCorrugatorSuperciliiLandmark => Color.red,

            FaceMuscleAnchorTag.BridgeOfTheNose => Color.blue,

            FaceMuscleAnchorTag.LLevatorLabiiSuperiorisLandmark => new Color(1f, 0.5f, 0.5f),
            FaceMuscleAnchorTag.RLevatorLabiiSuperiorisLandmark => new Color(1f, 0.5f, 0.5f),

            FaceMuscleAnchorTag.LDepressorLabiiInferiorisLandmark => new Color(0.5f, 0.8f, 1f),
            FaceMuscleAnchorTag.RDepressorLabiiInferiorisLandmark => new Color(0.5f, 0.8f, 1f),

            _ => _defaultColor
        };
    }

    private void CacheLipCenterIndices()
    {
        if (_calibrationData == null) return;

        var descriptions = _calibrationData.GetFaceLandmarkDescriptions();
        if (descriptions == null) return;

        for (int i = 0; i < descriptions.Count; i++)
        {
            var tag = descriptions[i].faceLandMarkTag;
            if (tag == FaceLandMarkTag.UpperLipCenter) _upperLipCenterIdx = descriptions[i].idx;
            if (tag == FaceLandMarkTag.DownLipCenter)  _downLipCenterIdx  = descriptions[i].idx;
            if (tag == FaceLandMarkTag.LLipAngle)      _lLipAngleIdx      = descriptions[i].idx;
            if (tag == FaceLandMarkTag.RLipAngle)      _rLipAngleIdx      = descriptions[i].idx;
        }
    }

    private void DrawMouthCenter()
    {
        if (!_drawMouthCenter) return;
        if (_upperLipCenterIdx < 0 || _downLipCenterIdx < 0) return;
        if (_faceLandmarkerRunner == null || _debugCanvas == null) return;

        FaceLandmarkerResult result = _faceLandmarkerRunner.GetLatestResult();
        if (result.faceLandmarks == null || result.faceLandmarks.Count == 0) return;

        var landmarks = result.faceLandmarks[0].landmarks;
        if (!IsValidIndex(_upperLipCenterIdx, landmarks.Count) || !IsValidIndex(_downLipCenterIdx, landmarks.Count))
            return;

        Vector3 upper = ToVector3(landmarks[_upperLipCenterIdx]);
        Vector3 lower = ToVector3(landmarks[_downLipCenterIdx]);
        Vector3 center = (upper + lower) * 0.5f;
        Vector2 canvasPos = NormalizedToCanvasLocal(center);

        CreatePoint("MouthCenter", canvasPos, _mouthCenterPointSize, _mouthCenterColor);

        if (_drawLines)
        {
            if (_lLipAngleIdx >= 0 && IsValidIndex(_lLipAngleIdx, landmarks.Count))
            {
                Vector2 lCornerPos = NormalizedToCanvasLocal(ToVector3(landmarks[_lLipAngleIdx]));
                DrawUiLine("Line_LCorner_MouthCenter", lCornerPos, canvasPos, 2f, _mouthCenterColor);
            }

            if (_rLipAngleIdx >= 0 && IsValidIndex(_rLipAngleIdx, landmarks.Count))
            {
                Vector2 rCornerPos = NormalizedToCanvasLocal(ToVector3(landmarks[_rLipAngleIdx]));
                DrawUiLine("Line_RCorner_MouthCenter", rCornerPos, canvasPos, 2f, _mouthCenterColor);
            }
        }

        if (_drawLabels)
        {
            CreateLabel(
                "MouthCenterLabel",
                "MouthCenter",
                canvasPos + _labelOffset,
                _mouthCenterColor,
                _labelFontSize
            );
        }
    }

    private void ClearDrawnUi()
    {
        for (int i = 0; i < _spawnedUi.Count; i++)
        {
            if (_spawnedUi[i] != null)
                Destroy(_spawnedUi[i]);
        }

        _spawnedUi.Clear();
    }
}