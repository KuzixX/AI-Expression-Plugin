#if UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using FaceMuscle.FaceRigPipeline.Core;
using FaceRig.Core;
using FaceRig.Data;
using UnityEditor;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Editor
{
    public class AnimationCorrectorWindow : EditorWindow
    {
        [SerializeField] private FaceRigRecording _recording;

        private int            _selectedClipIdx;
        private int            _currentFrame;
        private int            _lastLoadedFrame = -1;
        private Vector2        _scrollPos;
        private FaceRigContext _previewCtx;
        private bool           _autoPreview = true;
        private string[]       _clipNames;

        private bool _foldMouth = true;
        private bool _foldBrows = true;
        private bool _foldEyes  = true;
        private bool _foldJaw   = true;
        private bool _foldIris  = true;

        private int _frameMultiplier = 2;

        // Range Apply
        private bool _rangeMode;
        private int  _rangeFrom;
        private int  _rangeTo;

        // Copy-Paste
        private FrameSnapshot? _copiedFrame;

        // Curves View
        private bool   _showCurves;
        private int    _curveGroupIdx;
        private float  _curveHeight = 200f;
        private int    _selectedCurveChannel = -1; // index within current group
        private bool   _isDraggingCurve;
        private Vector2 _curvesScroll;

        // Tabs
        private int _tabIndex; // 0 = Edit, 1 = Export
        private static readonly string[] TabNames = { "Edit", "Export" };

        private static readonly string[] CurveGroupNames = { "Mouth", "Brows", "Eyes", "Jaw", "Iris X", "Iris Y" };

        private static readonly Color[] CurveColors =
        {
            new(0.2f, 0.8f, 0.2f), new(0.8f, 0.2f, 0.2f), new(0.2f, 0.5f, 1f),
            new(1f, 0.8f, 0.1f), new(1f, 0.4f, 0.7f), new(0.5f, 1f, 0.8f),
            new(0.9f, 0.5f, 0.1f), new(0.6f, 0.3f, 0.9f), new(0.3f, 0.9f, 0.5f),
            new(0.8f, 0.8f, 0.3f), new(0.4f, 0.7f, 0.7f), new(0.9f, 0.3f, 0.5f),
            new(0.5f, 0.5f, 1f),
        };

        // Playback
        private bool   _isPlaying;
        private bool   _loopPlayback;
        private double _lastEditorTime;
        private float  _playbackSpeed = 1f;
        private float  _playbackTime;  // accumulated clip time

        // Reference image
        private Texture2D _referenceTexture;
        private string    _lastRefClipPath;

        // Группировка мышц
        private static readonly HashSet<FaceMuscleAnchorTag> MouthTags = new()
        {
            FaceMuscleAnchorTag.lZygomaticusMajorLandmark,
            FaceMuscleAnchorTag.rZygomaticusMajorLandmark,
            FaceMuscleAnchorTag.lZygomaticusMinorLandmark,
            FaceMuscleAnchorTag.rZygomaticusMinorLandmark,
            FaceMuscleAnchorTag.lDepressorAnguliOrisLandmark,
            FaceMuscleAnchorTag.rDepressorAnguliOrisLandmark,
            FaceMuscleAnchorTag.lRisoriusLandmark,
            FaceMuscleAnchorTag.rRisoriusLandmark,
            FaceMuscleAnchorTag.LLevatorLabiiSuperiorisLandmark,
            FaceMuscleAnchorTag.RLevatorLabiiSuperiorisLandmark,
            FaceMuscleAnchorTag.LDepressorLabiiInferiorisLandmark,
            FaceMuscleAnchorTag.RDepressorLabiiInferiorisLandmark,
            FaceMuscleAnchorTag.OrbicularisOrisCenter,
        };

        private static readonly HashSet<FaceMuscleAnchorTag> BrowTags = new()
        {
            FaceMuscleAnchorTag.lFrontalisInsideLandmark,
            FaceMuscleAnchorTag.rFrontalisInsideLandmark,
            FaceMuscleAnchorTag.lFrontalisOuterLandmark,
            FaceMuscleAnchorTag.rFrontalisOuterLandmark,
            FaceMuscleAnchorTag.LCorrugatorSuperciliiLandmark,
            FaceMuscleAnchorTag.RCorrugatorSuperciliiLandmark,
        };

        private static readonly HashSet<FaceMuscleAnchorTag> EyeTags = new()
        {
            FaceMuscleAnchorTag.LOrbicularisOculi,
            FaceMuscleAnchorTag.ROrbicularisOculi,
        };

        private static readonly HashSet<FaceMuscleAnchorTag> JawTags = new()
        {
            FaceMuscleAnchorTag.BridgeOfTheNose,
            FaceMuscleAnchorTag.JawSlide,
        };

        [MenuItem("FaceRig/Animation Corrector")]
        public static void Open()
        {
            var window = GetWindow<AnimationCorrectorWindow>("Animation Corrector");
            window.minSize = new Vector2(620, 550);
        }

        private void OnEnable()
        {
            _previewCtx = new FaceRigContext();
            RefreshClipNames();
            EditorApplication.update += EditorUpdate;
        }

        private void OnDisable()
        {
            EditorApplication.update -= EditorUpdate;
            _isPlaying = false;

            if (_referenceTexture != null)
                DestroyImmediate(_referenceTexture);
        }

        private void EditorUpdate()
        {
            if (!_isPlaying || _recording == null || _recording.clips.Count == 0) return;

            var clip = _recording.clips[_selectedClipIdx];
            if (clip.frames.Count < 2) { _isPlaying = false; return; }

            double now = EditorApplication.timeSinceStartup;
            double dt = now - _lastEditorTime;
            _lastEditorTime = now;

            _playbackTime += (float)(dt * _playbackSpeed);

            float duration = clip.frames[clip.frames.Count - 1].timestamp;

            if (_playbackTime >= duration)
            {
                if (_loopPlayback)
                {
                    _playbackTime %= duration;
                }
                else
                {
                    _playbackTime = duration;
                    _isPlaying = false;
                }
            }

            // Binary search for frame at _playbackTime
            int lo = 0, hi = clip.frames.Count - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (clip.frames[mid].timestamp < _playbackTime)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            if (lo != _currentFrame)
            {
                _currentFrame = lo;
                PreviewFrame(clip);
                Repaint();
            }
        }

        private void OnGUI()
        {
            DrawRecordingField();

            if (_recording == null || _recording.clips.Count == 0)
            {
                EditorGUILayout.HelpBox("Assign a FaceRigRecording with clips.", MessageType.Info);
                return;
            }

            DrawClipSelector();

            var clip = _recording.clips[_selectedClipIdx];
            if (clip.frames.Count == 0)
            {
                EditorGUILayout.HelpBox("Selected clip has no frames.", MessageType.Warning);
                return;
            }

            DrawFrameScrubber(clip);

            // ── Tabs ──
            EditorGUILayout.Space(4);
            _tabIndex = GUILayout.Toolbar(_tabIndex, TabNames, EditorStyles.toolbarButton);
            EditorGUILayout.Space(4);

            if (_tabIndex == 0)
                DrawEditTab(clip);
            else
                DrawExportTab(clip);
        }

        // ─────────────────────── Reference Image ───────────────────────

        private void DrawReferenceImage(RecordingClip clip)
        {
            bool hasRef = !string.IsNullOrEmpty(clip.referenceFramesPath);

            EditorGUILayout.BeginVertical(GUILayout.Width(200));

            if (hasRef)
            {
                LoadReferenceFrame(clip);

                if (_referenceTexture != null)
                {
                    EditorGUILayout.LabelField("Reference", EditorStyles.boldLabel);
                    float aspect = (float)_referenceTexture.height / _referenceTexture.width;
                    float width = 190f;
                    float height = width * aspect;
                    var rect = GUILayoutUtility.GetRect(width, height);
                    GUI.DrawTexture(rect, _referenceTexture, ScaleMode.ScaleToFit);
                }
                else
                {
                    EditorGUILayout.HelpBox($"No image for frame {_currentFrame}", MessageType.None);
                }
            }
            else
            {
                EditorGUILayout.HelpBox("No reference frames recorded for this clip.", MessageType.None);
            }

            EditorGUILayout.EndVertical();
        }

        private void LoadReferenceFrame(RecordingClip clip)
        {
            if (_lastLoadedFrame == _currentFrame && _lastRefClipPath == clip.referenceFramesPath)
                return;

            _lastLoadedFrame = _currentFrame;
            _lastRefClipPath = clip.referenceFramesPath;

            // Convert Assets/... path to absolute
            string absDir = Path.Combine(Application.dataPath, "..", clip.referenceFramesPath);
            string filePath = Path.Combine(absDir, $"frame_{_currentFrame:D4}.jpg");

            if (!File.Exists(filePath))
            {
                if (_referenceTexture != null)
                    DestroyImmediate(_referenceTexture);
                _referenceTexture = null;
                return;
            }

            var bytes = File.ReadAllBytes(filePath);
            if (_referenceTexture == null)
                _referenceTexture = new Texture2D(2, 2);
            _referenceTexture.LoadImage(bytes);
        }

        // ─────────────────────── Recording & Clip ───────────────────────

        private void DrawRecordingField()
        {
            EditorGUI.BeginChangeCheck();
            _recording = (FaceRigRecording)EditorGUILayout.ObjectField(
                "Recording", _recording, typeof(FaceRigRecording), false);
            if (EditorGUI.EndChangeCheck())
            {
                _selectedClipIdx = 0;
                _currentFrame = 0;
                _lastLoadedFrame = -1;
                RefreshClipNames();
            }
        }

        private void DrawClipSelector()
        {
            if (_clipNames == null || _clipNames.Length != _recording.clips.Count)
                RefreshClipNames();

            EditorGUI.BeginChangeCheck();
            _selectedClipIdx = EditorGUILayout.Popup("Clip", _selectedClipIdx, _clipNames);
            if (EditorGUI.EndChangeCheck())
            {
                _currentFrame = 0;
                _lastLoadedFrame = -1;
            }
        }

        private void RefreshClipNames()
        {
            if (_recording == null || _recording.clips.Count == 0)
            {
                _clipNames = new string[0];
                return;
            }

            _clipNames = new string[_recording.clips.Count];
            for (int i = 0; i < _recording.clips.Count; i++)
                _clipNames[i] = string.IsNullOrEmpty(_recording.clips[i].name)
                    ? $"clip_{i}"
                    : _recording.clips[i].name;
        }

        // ─────────────────────── Frame Scrubber ───────────────────────

        private void DrawFrameScrubber(RecordingClip clip)
        {
            int maxFrame = clip.frames.Count - 1;
            var snapshot = clip.frames[_currentFrame];

            EditorGUILayout.LabelField(
                $"Frame {_currentFrame}/{maxFrame}   t = {snapshot.timestamp:F3}s",
                EditorStyles.miniLabel);

            EditorGUI.BeginChangeCheck();
            _currentFrame = EditorGUILayout.IntSlider("Frame", _currentFrame, 0, maxFrame);
            if (EditorGUI.EndChangeCheck() && _autoPreview)
                PreviewFrame(clip);

            // ── Playback controls ──
            EditorGUILayout.BeginHorizontal();

            if (_isPlaying)
            {
                if (GUILayout.Button("||  Pause", EditorStyles.miniButtonLeft, GUILayout.Width(70)))
                    _isPlaying = false;
            }
            else
            {
                if (GUILayout.Button(">>  Play", EditorStyles.miniButtonLeft, GUILayout.Width(70)))
                {
                    if (_currentFrame >= maxFrame) _currentFrame = 0;
                    _playbackTime = clip.frames[_currentFrame].timestamp;
                    _lastEditorTime = EditorApplication.timeSinceStartup;
                    _isPlaying = true;
                }
            }

            if (GUILayout.Button("|<  Reset", EditorStyles.miniButtonMid, GUILayout.Width(70)))
            {
                _isPlaying = false;
                _currentFrame = 0;
                _lastLoadedFrame = -1;
                if (_autoPreview) PreviewFrame(clip);
            }

            _loopPlayback = GUILayout.Toggle(_loopPlayback, "Loop", EditorStyles.miniButtonRight, GUILayout.Width(50));

            EditorGUILayout.LabelField("Speed", GUILayout.Width(40));
            _playbackSpeed = EditorGUILayout.Slider(_playbackSpeed, 0.1f, 3f);

            EditorGUILayout.EndHorizontal();
        }

        // ─────────────────────── Edit Tab ───────────────────────

        private void DrawEditTab(RecordingClip clip)
        {
            // Row 1: Preview / Save
            EditorGUILayout.BeginHorizontal();

            _autoPreview = GUILayout.Toggle(_autoPreview, "Auto Preview", EditorStyles.miniButtonLeft);

            if (GUILayout.Button("Preview", EditorStyles.miniButtonMid))
                PreviewFrame(clip);

            if (GUILayout.Button("Save", EditorStyles.miniButtonRight))
            {
                EditorUtility.SetDirty(_recording);
                AssetDatabase.SaveAssets();
                Debug.Log("[AnimationCorrector] Saved.");
            }

            EditorGUILayout.EndHorizontal();

            // Row 2: Copy / Paste
            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Copy Frame", EditorStyles.miniButtonLeft, GUILayout.Width(90)))
            {
                _copiedFrame = clip.frames[_currentFrame];
                Debug.Log($"[AnimationCorrector] Copied frame {_currentFrame}");
            }

            GUI.enabled = _copiedFrame.HasValue;
            if (GUILayout.Button("Paste Frame", EditorStyles.miniButtonRight, GUILayout.Width(90)))
            {
                Undo.RecordObject(_recording, "Paste Frame");
                var pasted = _copiedFrame.Value;
                pasted.timestamp = clip.frames[_currentFrame].timestamp;
                clip.frames[_currentFrame] = pasted;
                EditorUtility.SetDirty(_recording);
                if (_autoPreview) PreviewFrame(clip);
                Debug.Log($"[AnimationCorrector] Pasted to frame {_currentFrame}");
            }
            GUI.enabled = true;

            EditorGUILayout.EndHorizontal();

            // ── Range Apply ──
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Range Apply", new GUIStyle(EditorStyles.boldLabel) { alignment = TextAnchor.MiddleCenter });

            EditorGUILayout.BeginHorizontal();

            int maxFrame = clip.frames.Count - 1;
            _rangeMode = GUILayout.Toggle(_rangeMode, "Enable", EditorStyles.miniButton, GUILayout.Width(60));

            if (_rangeMode)
            {
                EditorGUILayout.LabelField("From", GUILayout.Width(32));
                _rangeFrom = EditorGUILayout.IntField(_rangeFrom, GUILayout.Width(45));
                EditorGUILayout.LabelField("To", GUILayout.Width(18));
                _rangeTo = EditorGUILayout.IntField(_rangeTo, GUILayout.Width(45));

                _rangeFrom = Mathf.Clamp(_rangeFrom, 0, maxFrame);
                _rangeTo   = Mathf.Clamp(_rangeTo, _rangeFrom, maxFrame);

                if (GUILayout.Button("Set From Here", EditorStyles.miniButton, GUILayout.Width(85)))
                    _rangeFrom = _currentFrame;
                if (GUILayout.Button("Set To Here", EditorStyles.miniButton, GUILayout.Width(75)))
                    _rangeTo = _currentFrame;

                if (GUILayout.Button("Apply to Range", EditorStyles.miniButton, GUILayout.Width(95)))
                {
                    Undo.RecordObject(_recording, "Apply to Range");
                    ApplyToRange(clip, _rangeFrom, _rangeTo, _currentFrame);
                    EditorUtility.SetDirty(_recording);
                }
            }

            EditorGUILayout.EndHorizontal();

            // ── Interpolation ──
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Interpolation", new GUIStyle(EditorStyles.boldLabel) { alignment = TextAnchor.MiddleCenter });

            EditorGUILayout.BeginHorizontal();

            EditorGUILayout.LabelField("Multiply", GUILayout.Width(52));
            _frameMultiplier = EditorGUILayout.IntField(_frameMultiplier, GUILayout.Width(32));
            _frameMultiplier = Mathf.Clamp(_frameMultiplier, 2, 16);
            EditorGUILayout.LabelField($"x  ({clip.frames.Count} → {clip.frames.Count + (clip.frames.Count - 1) * (_frameMultiplier - 1)} frames)",
                EditorStyles.miniLabel, GUILayout.Width(160));

            if (GUILayout.Button($"Multiply x{_frameMultiplier}", EditorStyles.miniButton))
            {
                Undo.RecordObject(_recording, "Multiply Frames");
                MultiplyFrames(clip, _frameMultiplier);
                EditorUtility.SetDirty(_recording);
            }

            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(4);

            // ── View toggle: Sliders / Curves ──
            EditorGUILayout.BeginHorizontal();
            _showCurves = GUILayout.Toggle(_showCurves, _showCurves ? "Curves View" : "Sliders View",
                EditorStyles.toolbarButton, GUILayout.Width(100));
            if (_showCurves)
            {
                _curveGroupIdx = EditorGUILayout.Popup(_curveGroupIdx, CurveGroupNames, GUILayout.Width(100));
                EditorGUILayout.LabelField("Height", GUILayout.Width(40));
                _curveHeight = EditorGUILayout.Slider(_curveHeight, 100f, 500f);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(4);

            if (_showCurves)
            {
                bool curveChanged = DrawCurvesView(clip);
                if (curveChanged)
                {
                    EditorUtility.SetDirty(_recording);
                    if (_autoPreview) PreviewFrame(clip);
                }
            }
            else
            {
                // ── Main layout: Reference image LEFT | Sliders RIGHT ──
                EditorGUILayout.BeginHorizontal();
                DrawReferenceImage(clip);

                EditorGUILayout.BeginVertical();
                _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos);
                bool changed = false;

                changed |= DrawGroupedActivations(clip, "Mouth", ref _foldMouth, MouthTags);
                changed |= DrawGroupedActivations(clip, "Brows", ref _foldBrows, BrowTags);
                changed |= DrawGroupedActivations(clip, "Eyes (Eyelids)", ref _foldEyes, EyeTags);
                changed |= DrawJawStrainSliders(clip);
                changed |= DrawIrisSliders(clip);

                EditorGUILayout.EndScrollView();
                EditorGUILayout.EndVertical();
                EditorGUILayout.EndHorizontal();

                if (changed)
                {
                    EditorUtility.SetDirty(_recording);
                    if (_autoPreview) PreviewFrame(clip);
                }
            }
        }

        // ─────────────────────── Export Tab ───────────────────────

        private void DrawExportTab(RecordingClip clip)
        {
            EditorGUILayout.LabelField("Export Current Clip", EditorStyles.boldLabel);

            if (GUILayout.Button("Export CSV (current clip)", GUILayout.Height(28)))
                ExportCSV(clip);

            EditorGUILayout.Space(12);
            EditorGUILayout.LabelField("Export All Clips", EditorStyles.boldLabel);

            if (GUILayout.Button("Export All (each clip → separate file)", GUILayout.Height(28)))
                ExportAllCSV();

            EditorGUILayout.Space(4);

            if (GUILayout.Button("Merge & Export (all clips → one file)", GUILayout.Height(28)))
                ExportMergedCSV();

            EditorGUILayout.Space(12);

            // Info
            EditorGUILayout.HelpBox(
                $"Recording: {_recording.clips.Count} clips, " +
                $"{_recording.clips.Sum(c => c.frames.Count)} total frames\n\n" +
                "Export CSV — текущий клип в отдельный файл\n" +
                "Export All — каждый клип в отдельный файл\n" +
                "Merge & Export — все клипы в один файл для обучения",
                MessageType.Info);
        }

        // ─────────────────────── Grouped Activations ───────────────────────

        private bool DrawGroupedActivations(RecordingClip clip, string header, ref bool foldout,
            HashSet<FaceMuscleAnchorTag> filter)
        {
            var snapshot = clip.frames[_currentFrame];
            if (snapshot.activations == null || snapshot.activations.Length == 0) return false;

            foldout = EditorGUILayout.Foldout(foldout, header, true, EditorStyles.foldoutHeader);
            if (!foldout) return false;

            EditorGUI.indentLevel++;
            bool changed = false;

            for (int i = 0; i < snapshot.activations.Length; i++)
            {
                var act = snapshot.activations[i];
                if (!filter.Contains(act.tag)) continue;

                EditorGUI.BeginChangeCheck();
                float newVal = EditorGUILayout.Slider(act.tag.ToString(), act.value, 0f, 1f);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(_recording, "Edit Activation");
                    act.value = newVal;
                    snapshot.activations[i] = act;
                    clip.frames[_currentFrame] = snapshot;
                    changed = true;
                }
            }

            EditorGUI.indentLevel--;
            return changed;
        }

        // ─────────────────────── Jaw Strain Sliders ───────────────────────

        private bool DrawJawStrainSliders(RecordingClip clip)
        {
            var snapshot = clip.frames[_currentFrame];

            _foldJaw = EditorGUILayout.Foldout(_foldJaw, "Jaw", true, EditorStyles.foldoutHeader);
            if (!_foldJaw) return false;

            EditorGUI.indentLevel++;
            bool changed = false;

            foreach (var tag in JawTags)
            {
                // Find existing strain index, or -1
                int idx = -1;
                float val = 0f;
                if (snapshot.strains != null)
                {
                    for (int i = 0; i < snapshot.strains.Length; i++)
                    {
                        if (snapshot.strains[i].tag == tag) { idx = i; val = snapshot.strains[i].value; break; }
                    }
                }

                EditorGUI.BeginChangeCheck();
                float newVal = EditorGUILayout.Slider(tag.ToString(), val, -1f, 1f);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(_recording, "Edit Jaw Strain");
                    if (idx >= 0)
                    {
                        var s = snapshot.strains[idx];
                        s.value = newVal;
                        snapshot.strains[idx] = s;
                    }
                    else
                    {
                        var list = snapshot.strains != null
                            ? new List<TagFloat>(snapshot.strains)
                            : new List<TagFloat>();
                        list.Add(new TagFloat { tag = tag, value = newVal });
                        snapshot.strains = list.ToArray();
                    }
                    clip.frames[_currentFrame] = snapshot;
                    changed = true;
                }
            }

            EditorGUI.indentLevel--;
            return changed;
        }

        // ─────────────────────── Iris Sliders ───────────────────────

        private bool DrawIrisSliders(RecordingClip clip)
        {
            var snapshot = clip.frames[_currentFrame];
            if (snapshot.irisPositions == null || snapshot.irisPositions.Length == 0) return false;

            _foldIris = EditorGUILayout.Foldout(_foldIris, "Eyes (Iris Direction)", true, EditorStyles.foldoutHeader);
            if (!_foldIris) return false;

            EditorGUI.indentLevel++;
            bool changed = false;

            for (int i = 0; i < snapshot.irisPositions.Length; i++)
            {
                var iris = snapshot.irisPositions[i];

                EditorGUILayout.LabelField(iris.tag.ToString(), EditorStyles.boldLabel);
                EditorGUI.indentLevel++;

                EditorGUI.BeginChangeCheck();
                float newX = EditorGUILayout.Slider("Horizontal", iris.value.x, -1f, 1f);
                float newY = EditorGUILayout.Slider("Vertical",   iris.value.y, -1f, 1f);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(_recording, "Edit Iris");
                    iris.value = new Vector2(newX, newY);
                    snapshot.irisPositions[i] = iris;
                    clip.frames[_currentFrame] = snapshot;
                    changed = true;
                }

                EditorGUI.indentLevel--;
            }

            EditorGUI.indentLevel--;
            return changed;
        }

        // ─────────────────────── Multiply Frames ───────────────────────

        private void MultiplyFrames(RecordingClip clip, int multiplier)
        {
            int originalCount = clip.frames.Count;
            if (originalCount < 2) return;

            var newFrames = new List<FrameSnapshot>(originalCount * multiplier);

            for (int i = 0; i < originalCount - 1; i++)
            {
                var a = clip.frames[i];
                var b = clip.frames[i + 1];

                newFrames.Add(a); // original frame

                for (int s = 1; s < multiplier; s++)
                {
                    float t = (float)s / multiplier;
                    newFrames.Add(new FrameSnapshot
                    {
                        timestamp     = Mathf.Lerp(a.timestamp, b.timestamp, t),
                        activations   = LerpTagFloats(a.activations, b.activations, t),
                        strains       = LerpTagFloats(a.strains, b.strains, t),
                        irisPositions = LerpTagVector2s(a.irisPositions, b.irisPositions, t),
                    });
                }
            }

            newFrames.Add(clip.frames[originalCount - 1]); // last original frame

            clip.frames.Clear();
            clip.frames.AddRange(newFrames);

            _currentFrame = 0;
            _lastLoadedFrame = -1;

            Debug.Log($"[AnimationCorrector] Multiplied x{multiplier}: {originalCount} → {clip.frames.Count} frames");
        }

        private static TagFloat[] LerpTagFloats(TagFloat[] a, TagFloat[] b, float t)
        {
            if (a == null || a.Length == 0) return b ?? System.Array.Empty<TagFloat>();
            if (b == null || b.Length == 0) return a;

            var bMap = new Dictionary<FaceMuscleAnchorTag, float>(b.Length);
            for (int i = 0; i < b.Length; i++)
                bMap[b[i].tag] = b[i].value;

            var result = new TagFloat[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                float bVal = bMap.TryGetValue(a[i].tag, out var v) ? v : a[i].value;
                result[i] = new TagFloat
                {
                    tag = a[i].tag,
                    value = Mathf.Lerp(a[i].value, bVal, t),
                };
            }
            return result;
        }

        private static TagVector2[] LerpTagVector2s(TagVector2[] a, TagVector2[] b, float t)
        {
            if (a == null || a.Length == 0) return b ?? System.Array.Empty<TagVector2>();
            if (b == null || b.Length == 0) return a;

            var bMap = new Dictionary<FaceLandMarkTag, Vector2>(b.Length);
            for (int i = 0; i < b.Length; i++)
                bMap[b[i].tag] = b[i].value;

            var result = new TagVector2[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                Vector2 bVal = bMap.TryGetValue(a[i].tag, out var v) ? v : a[i].value;
                result[i] = new TagVector2
                {
                    tag = a[i].tag,
                    value = Vector2.Lerp(a[i].value, bVal, t),
                };
            }
            return result;
        }

        // ─────────────────────── Curves View ───────────────────────

        private struct CurveChannel
        {
            public string name;
            public Color  color;
        }

        private List<CurveChannel> GetChannelsForGroup(RecordingClip clip)
        {
            var channels = new List<CurveChannel>();
            var first = clip.frames[0];

            HashSet<FaceMuscleAnchorTag> filter = _curveGroupIdx switch
            {
                0 => MouthTags,
                1 => BrowTags,
                2 => EyeTags,
                3 => JawTags,
                _ => null
            };

            if (_curveGroupIdx <= 3)
            {
                // Activations or Strains
                var source = _curveGroupIdx == 3 ? first.strains : first.activations;
                if (source == null) return channels;

                int ci = 0;
                for (int i = 0; i < source.Length; i++)
                {
                    if (filter != null && !filter.Contains(source[i].tag)) continue;
                    channels.Add(new CurveChannel
                    {
                        name  = source[i].tag.ToString(),
                        color = CurveColors[ci % CurveColors.Length],
                    });
                    ci++;
                }
            }
            else
            {
                // Iris X or Y
                if (first.irisPositions == null) return channels;
                for (int i = 0; i < first.irisPositions.Length; i++)
                {
                    string axis = _curveGroupIdx == 4 ? "X" : "Y";
                    channels.Add(new CurveChannel
                    {
                        name  = $"{first.irisPositions[i].tag} {axis}",
                        color = CurveColors[i % CurveColors.Length],
                    });
                }
            }
            return channels;
        }

        private float GetChannelValue(FrameSnapshot snap, int channelIdx)
        {
            HashSet<FaceMuscleAnchorTag> filter = _curveGroupIdx switch
            {
                0 => MouthTags,
                1 => BrowTags,
                2 => EyeTags,
                3 => JawTags,
                _ => null
            };

            if (_curveGroupIdx <= 2)
            {
                // Activations
                if (snap.activations == null) return 0f;
                int ci = 0;
                for (int i = 0; i < snap.activations.Length; i++)
                {
                    if (filter != null && !filter.Contains(snap.activations[i].tag)) continue;
                    if (ci == channelIdx) return snap.activations[i].value;
                    ci++;
                }
            }
            else if (_curveGroupIdx == 3)
            {
                // Strains
                if (snap.strains == null) return 0f;
                int ci = 0;
                for (int i = 0; i < snap.strains.Length; i++)
                {
                    if (!JawTags.Contains(snap.strains[i].tag)) continue;
                    if (ci == channelIdx) return snap.strains[i].value;
                    ci++;
                }
            }
            else
            {
                // Iris
                if (snap.irisPositions == null) return 0f;
                if (channelIdx < snap.irisPositions.Length)
                {
                    var v = snap.irisPositions[channelIdx].value;
                    return _curveGroupIdx == 4 ? v.x : v.y;
                }
            }
            return 0f;
        }

        private void SetChannelValue(RecordingClip clip, int frameIdx, int channelIdx, float value)
        {
            var snap = clip.frames[frameIdx];

            HashSet<FaceMuscleAnchorTag> filter = _curveGroupIdx switch
            {
                0 => MouthTags,
                1 => BrowTags,
                2 => EyeTags,
                3 => JawTags,
                _ => null
            };

            if (_curveGroupIdx <= 2)
            {
                if (snap.activations == null) return;
                int ci = 0;
                for (int i = 0; i < snap.activations.Length; i++)
                {
                    if (filter != null && !filter.Contains(snap.activations[i].tag)) continue;
                    if (ci == channelIdx)
                    {
                        var a = snap.activations[i];
                        a.value = Mathf.Clamp01(value);
                        snap.activations[i] = a;
                        clip.frames[frameIdx] = snap;
                        return;
                    }
                    ci++;
                }
            }
            else if (_curveGroupIdx == 3)
            {
                if (snap.strains == null) return;
                int ci = 0;
                for (int i = 0; i < snap.strains.Length; i++)
                {
                    if (!JawTags.Contains(snap.strains[i].tag)) continue;
                    if (ci == channelIdx)
                    {
                        var s = snap.strains[i];
                        s.value = Mathf.Clamp(value, -1f, 1f);
                        snap.strains[i] = s;
                        clip.frames[frameIdx] = snap;
                        return;
                    }
                    ci++;
                }
            }
            else
            {
                if (snap.irisPositions == null || channelIdx >= snap.irisPositions.Length) return;
                var iris = snap.irisPositions[channelIdx];
                if (_curveGroupIdx == 4)
                    iris.value.x = Mathf.Clamp(value, -1f, 1f);
                else
                    iris.value.y = Mathf.Clamp(value, -1f, 1f);
                snap.irisPositions[channelIdx] = iris;
                clip.frames[frameIdx] = snap;
            }
        }

        private bool DrawCurvesView(RecordingClip clip)
        {
            var channels = GetChannelsForGroup(clip);
            if (channels.Count == 0)
            {
                EditorGUILayout.HelpBox("No data for this group.", MessageType.Info);
                return false;
            }

            bool changed = false;
            float minVal = _curveGroupIdx <= 2 ? 0f : -1f;
            float maxVal = 1f;

            // Legend (clickable to select channel)
            _curvesScroll = EditorGUILayout.BeginScrollView(_curvesScroll, GUILayout.Height(50));
            EditorGUILayout.BeginHorizontal();
            for (int c = 0; c < channels.Count; c++)
            {
                var prevColor = GUI.backgroundColor;
                if (_selectedCurveChannel == c)
                    GUI.backgroundColor = channels[c].color;

                if (GUILayout.Button(channels[c].name, EditorStyles.miniButton, GUILayout.MinWidth(60)))
                    _selectedCurveChannel = _selectedCurveChannel == c ? -1 : c;

                GUI.backgroundColor = prevColor;
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndScrollView();

            // Graph area
            var graphRect = GUILayoutUtility.GetRect(position.width - 20f, _curveHeight);
            EditorGUI.DrawRect(graphRect, new Color(0.15f, 0.15f, 0.15f));

            // Grid lines
            int gridLines = 4;
            for (int g = 0; g <= gridLines; g++)
            {
                float gy = graphRect.y + graphRect.height * g / gridLines;
                EditorGUI.DrawRect(new Rect(graphRect.x, gy, graphRect.width, 1), new Color(0.3f, 0.3f, 0.3f));

                float labelVal = Mathf.Lerp(maxVal, minVal, (float)g / gridLines);
                GUI.Label(new Rect(graphRect.x + 2, gy - 8, 40, 16), labelVal.ToString("F1"),
                    EditorStyles.miniLabel);
            }

            // Zero line for bipolar groups
            if (minVal < 0f)
            {
                float zeroY = graphRect.y + graphRect.height * (maxVal / (maxVal - minVal));
                EditorGUI.DrawRect(new Rect(graphRect.x, zeroY, graphRect.width, 1), new Color(0.5f, 0.5f, 0.5f));
            }

            // Current frame marker
            float frameX = graphRect.x + (graphRect.width * _currentFrame / Mathf.Max(1, clip.frames.Count - 1));
            EditorGUI.DrawRect(new Rect(frameX - 1, graphRect.y, 2, graphRect.height),
                new Color(1f, 1f, 1f, 0.6f));

            // Range markers
            if (_rangeMode)
            {
                float fromX = graphRect.x + graphRect.width * _rangeFrom / Mathf.Max(1, clip.frames.Count - 1);
                float toX   = graphRect.x + graphRect.width * _rangeTo / Mathf.Max(1, clip.frames.Count - 1);
                EditorGUI.DrawRect(new Rect(fromX, graphRect.y, toX - fromX, graphRect.height),
                    new Color(0.3f, 0.5f, 1f, 0.1f));
            }

            // Draw curves
            int frameCount = clip.frames.Count;
            // Downsample if too many frames
            int step = Mathf.Max(1, frameCount / (int)graphRect.width);

            for (int c = 0; c < channels.Count; c++)
            {
                Color lineColor = channels[c].color;
                float thickness = 1.5f;

                if (_selectedCurveChannel >= 0 && _selectedCurveChannel != c)
                {
                    lineColor.a = 0.2f; // dim non-selected
                    thickness = 1f;
                }
                else if (_selectedCurveChannel == c)
                {
                    thickness = 2.5f;
                }

                // Build points
                var points = new List<Vector3>();
                for (int f = 0; f < frameCount; f += step)
                {
                    float val = GetChannelValue(clip.frames[f], c);
                    float nx = (float)f / Mathf.Max(1, frameCount - 1);
                    float ny = (val - minVal) / (maxVal - minVal);

                    float px = graphRect.x + nx * graphRect.width;
                    float py = graphRect.y + graphRect.height * (1f - ny);
                    points.Add(new Vector3(px, py, 0));
                }

                // Make sure we include the last frame
                if ((frameCount - 1) % step != 0)
                {
                    float val = GetChannelValue(clip.frames[frameCount - 1], c);
                    float ny = (val - minVal) / (maxVal - minVal);
                    points.Add(new Vector3(graphRect.xMax, graphRect.y + graphRect.height * (1f - ny), 0));
                }

                Handles.color = lineColor;
                if (points.Count > 1)
                    Handles.DrawAAPolyLine(thickness, points.ToArray());
            }

            // Draw keyframe dot for selected channel at current frame
            if (_selectedCurveChannel >= 0 && _selectedCurveChannel < channels.Count)
            {
                float val = GetChannelValue(clip.frames[_currentFrame], _selectedCurveChannel);
                float ny = (val - minVal) / (maxVal - minVal);
                float dotX = frameX;
                float dotY = graphRect.y + graphRect.height * (1f - ny);

                var dotRect = new Rect(dotX - 5, dotY - 5, 10, 10);
                EditorGUI.DrawRect(dotRect, channels[_selectedCurveChannel].color);
                EditorGUI.DrawRect(new Rect(dotX - 3, dotY - 3, 6, 6), Color.white);
            }

            // Mouse interaction
            var evt = Event.current;
            if (graphRect.Contains(evt.mousePosition))
            {
                if (evt.type == EventType.MouseDown && evt.button == 0)
                {
                    float mx = (evt.mousePosition.x - graphRect.x) / graphRect.width;
                    _currentFrame = Mathf.Clamp(Mathf.RoundToInt(mx * (frameCount - 1)), 0, frameCount - 1);

                    if (_selectedCurveChannel >= 0)
                    {
                        Undo.RecordObject(_recording, "Edit Curve");
                        _isDraggingCurve = true;
                    }

                    if (_autoPreview) PreviewFrame(clip);
                    evt.Use();
                    Repaint();
                }
                else if (evt.type == EventType.MouseDrag && _isDraggingCurve && _selectedCurveChannel >= 0)
                {
                    // Drag to edit value
                    float mx = (evt.mousePosition.x - graphRect.x) / graphRect.width;
                    float my = 1f - (evt.mousePosition.y - graphRect.y) / graphRect.height;

                    _currentFrame = Mathf.Clamp(Mathf.RoundToInt(mx * (frameCount - 1)), 0, frameCount - 1);
                    float newVal = Mathf.Lerp(minVal, maxVal, my);

                    SetChannelValue(clip, _currentFrame, _selectedCurveChannel, newVal);
                    changed = true;

                    if (_autoPreview) PreviewFrame(clip);
                    evt.Use();
                    Repaint();
                }
                else if (evt.type == EventType.MouseUp)
                {
                    _isDraggingCurve = false;
                    evt.Use();
                }

                // Scroll to zoom height
                if (evt.type == EventType.ScrollWheel)
                {
                    _curveHeight = Mathf.Clamp(_curveHeight - evt.delta.y * 10f, 100f, 500f);
                    evt.Use();
                    Repaint();
                }
            }

            if (evt.type == EventType.MouseUp)
                _isDraggingCurve = false;

            // Value label
            if (_selectedCurveChannel >= 0 && _selectedCurveChannel < channels.Count)
            {
                float val = GetChannelValue(clip.frames[_currentFrame], _selectedCurveChannel);
                EditorGUILayout.LabelField(
                    $"{channels[_selectedCurveChannel].name}: {val:F3} (frame {_currentFrame})",
                    EditorStyles.boldLabel);
            }

            return changed;
        }

        // ─────────────────────── Export CSV ───────────────────────

        private void ExportCSV(RecordingClip clip)
        {
            string dir = EditorUtility.SaveFolderPanel("Export CSV to folder", Application.dataPath, "");
            if (string.IsNullOrEmpty(dir)) return;

            string csv = BuildCSV(clip);
            string clipName = string.IsNullOrEmpty(clip.name) ? "clip" : clip.name;
            string path = Path.Combine(dir, $"{clipName}.csv");
            File.WriteAllText(path, csv);
            Debug.Log($"[AnimationCorrector] Exported {clip.frames.Count} frames → {path}");
            EditorUtility.RevealInFinder(path);
        }

        // ─────────────────────── Export All / Merge ───────────────────────

        private void ExportAllCSV()
        {
            string dir = EditorUtility.SaveFolderPanel("Export all clips to folder", Application.dataPath, "");
            if (string.IsNullOrEmpty(dir)) return;

            int total = 0;
            foreach (var clip in _recording.clips)
            {
                if (clip.frames.Count == 0) continue;
                string csv = BuildCSV(clip);
                string clipName = string.IsNullOrEmpty(clip.name) ? $"clip_{total}" : clip.name;
                string path = Path.Combine(dir, $"{clipName}.csv");
                File.WriteAllText(path, csv);
                total++;
            }

            Debug.Log($"[AnimationCorrector] Exported {total} clips → {dir}");
            EditorUtility.RevealInFinder(dir);
        }

        private void ExportMergedCSV()
        {
            string path = EditorUtility.SaveFilePanel("Export merged CSV", Application.dataPath, "merged", "csv");
            if (string.IsNullOrEmpty(path)) return;

            // Collect all tags across all clips
            var actTags  = new List<FaceMuscleAnchorTag>();
            var strTags  = new List<FaceMuscleAnchorTag>();
            var irisTags = new List<FaceLandMarkTag>();

            foreach (var clip in _recording.clips)
                CollectTags(clip, actTags, strTags, irisTags);

            string header = BuildHeader(actTags, strTags, irisTags);

            var sb = new StringBuilder();
            sb.AppendLine(header);

            int totalFrames = 0;
            foreach (var clip in _recording.clips)
            {
                if (clip.frames.Count == 0) continue;
                foreach (var frame in clip.frames)
                {
                    sb.AppendLine(BuildRow(frame, actTags, strTags, irisTags));
                    totalFrames++;
                }
            }

            File.WriteAllText(path, sb.ToString());
            Debug.Log($"[AnimationCorrector] Merged {_recording.clips.Count} clips, {totalFrames} frames → {path}");
            EditorUtility.RevealInFinder(path);
        }

        // ─────────────────────── CSV Helpers ───────────────────────

        private string BuildCSV(RecordingClip clip)
        {
            var actTags  = new List<FaceMuscleAnchorTag>();
            var strTags  = new List<FaceMuscleAnchorTag>();
            var irisTags = new List<FaceLandMarkTag>();
            CollectTags(clip, actTags, strTags, irisTags);

            var sb = new StringBuilder();
            sb.AppendLine(BuildHeader(actTags, strTags, irisTags));

            foreach (var frame in clip.frames)
                sb.AppendLine(BuildRow(frame, actTags, strTags, irisTags));

            return sb.ToString();
        }

        private void CollectTags(RecordingClip clip,
            List<FaceMuscleAnchorTag> actTags,
            List<FaceMuscleAnchorTag> strTags,
            List<FaceLandMarkTag> irisTags)
        {
            foreach (var frame in clip.frames)
            {
                if (frame.activations != null)
                    foreach (var a in frame.activations)
                        if (!actTags.Contains(a.tag)) actTags.Add(a.tag);
                if (frame.strains != null)
                    foreach (var s in frame.strains)
                        if (!strTags.Contains(s.tag)) strTags.Add(s.tag);
                if (frame.irisPositions != null)
                    foreach (var ip in frame.irisPositions)
                        if (!irisTags.Contains(ip.tag)) irisTags.Add(ip.tag);
            }
        }

        private string BuildHeader(
            List<FaceMuscleAnchorTag> actTags,
            List<FaceMuscleAnchorTag> strTags,
            List<FaceLandMarkTag> irisTags)
        {
            var sb = new StringBuilder();
            sb.Append("timestamp");
            foreach (var t in actTags)  sb.Append($",act_{t}");
            foreach (var t in strTags)  sb.Append($",str_{t}");
            foreach (var t in irisTags) { sb.Append($",iris_{t}_x"); sb.Append($",iris_{t}_y"); }
            return sb.ToString();
        }

        private string BuildRow(FrameSnapshot frame,
            List<FaceMuscleAnchorTag> actTags,
            List<FaceMuscleAnchorTag> strTags,
            List<FaceLandMarkTag> irisTags)
        {
            var sb = new StringBuilder();
            sb.Append(frame.timestamp.ToString("F4"));

            var actMap = new Dictionary<FaceMuscleAnchorTag, float>();
            if (frame.activations != null)
                foreach (var a in frame.activations) actMap[a.tag] = a.value;
            foreach (var t in actTags)
                sb.Append($",{(actMap.TryGetValue(t, out var v) ? v : 0f).ToString("F5")}");

            var strMap = new Dictionary<FaceMuscleAnchorTag, float>();
            if (frame.strains != null)
                foreach (var s in frame.strains) strMap[s.tag] = s.value;
            foreach (var t in strTags)
                sb.Append($",{(strMap.TryGetValue(t, out var v) ? v : 0f).ToString("F5")}");

            var irisMap = new Dictionary<FaceLandMarkTag, Vector2>();
            if (frame.irisPositions != null)
                foreach (var ip in frame.irisPositions) irisMap[ip.tag] = ip.value;
            foreach (var t in irisTags)
            {
                var val = irisMap.TryGetValue(t, out var iv) ? iv : Vector2.zero;
                sb.Append($",{val.x:F5},{val.y:F5}");
            }

            return sb.ToString();
        }

        // ─────────────────────── Range Apply ───────────────────────

        /// <summary>
        /// Takes current frame's values and interpolates them across [rangeFrom..rangeTo].
        /// Edges (rangeFrom, rangeTo) keep their original values, frames in between
        /// blend between edge values and the edited frame proportionally.
        /// </summary>
        private void ApplyToRange(RecordingClip clip, int rangeFrom, int rangeTo, int editedFrame)
        {
            if (rangeFrom >= rangeTo) return;

            editedFrame = Mathf.Clamp(editedFrame, rangeFrom, rangeTo);

            var edited = clip.frames[editedFrame];
            var first  = clip.frames[rangeFrom];
            var last   = clip.frames[rangeTo];

            for (int i = rangeFrom; i <= rangeTo; i++)
            {
                if (i == editedFrame) continue;

                float t;
                TagFloat[] baseAct, baseStr;
                TagVector2[] baseIris;

                if (i <= editedFrame)
                {
                    t = (editedFrame - rangeFrom) == 0 ? 1f : (float)(i - rangeFrom) / (editedFrame - rangeFrom);
                    baseAct  = first.activations;
                    baseStr  = first.strains;
                    baseIris = first.irisPositions;
                }
                else
                {
                    t = (rangeTo - editedFrame) == 0 ? 0f : (float)(i - editedFrame) / (rangeTo - editedFrame);
                    baseAct  = edited.activations;
                    baseStr  = edited.strains;
                    baseIris = edited.irisPositions;
                }

                var targetAct  = i <= editedFrame ? edited.activations : last.activations;
                var targetStr  = i <= editedFrame ? edited.strains : last.strains;
                var targetIris = i <= editedFrame ? edited.irisPositions : last.irisPositions;

                var snap = clip.frames[i];
                snap.activations   = LerpTagFloats(baseAct, targetAct, t);
                snap.strains       = LerpTagFloats(baseStr, targetStr, t);
                snap.irisPositions = LerpTagVector2s(baseIris, targetIris, t);
                clip.frames[i] = snap;
            }

            Debug.Log($"[AnimationCorrector] Applied range [{rangeFrom}..{rangeTo}] from frame {editedFrame}");
        }

        // ─────────────────────── Preview ───────────────────────

        private void PreviewFrame(RecordingClip clip)
        {
            var snapshot = clip.frames[_currentFrame];

            _previewCtx.Clear();

            if (snapshot.activations != null)
                for (int i = 0; i < snapshot.activations.Length; i++)
                    _previewCtx.Activations[snapshot.activations[i].tag] = snapshot.activations[i].value;

            // Ensure all JawTags exist with default 0 (for old recordings missing JawSlide etc.)
            foreach (var tag in JawTags)
                _previewCtx.Strains[tag] = 0f;

            if (snapshot.strains != null)
                for (int i = 0; i < snapshot.strains.Length; i++)
                    _previewCtx.Strains[snapshot.strains[i].tag] = snapshot.strains[i].value;

            if (snapshot.irisPositions != null)
                for (int i = 0; i < snapshot.irisPositions.Length; i++)
                    _previewCtx.IrisPositions[snapshot.irisPositions[i].tag] = snapshot.irisPositions[i].value;

            var drivers = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
            foreach (var mb in drivers)
            {
                if (mb is IPipelineStep step && mb is not StartUp)
                    step.Execute(_previewCtx);
            }

            SceneView.RepaintAll();
        }
    }
}
#endif
