using System.IO;
using System.Linq;
using FaceRig.Core;
using Mediapipe.Unity;
using Mediapipe.Unity.Sample;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Record")]
    public class RecordStepAsset : PipelineStepAsset
    {
        [SerializeField] private FaceRigRecording _recording;
        [SerializeField] [Range(1, 120)] private int _targetFps = 30;
        [SerializeField] private bool _captureReference = true;
        [SerializeField] [Range(32, 512)] private int _referenceSize = 128;

        public override IPipelineStep CreateStep() => new RecordStep(_recording, _targetFps, _captureReference, _referenceSize);
    }

    public class RecordStep : IPipelineStep
    {
        private readonly FaceRigRecording _recording;
        private readonly float           _minInterval;
        private readonly bool            _captureReference;
        private readonly int             _referenceSize;
        private bool           _isRecording;
        private float          _recordStartTime;
        private float          _lastSnapshotTime;
        private RecordingClip  _currentClip;
        private string         _framesDir;
        private int            _frameCounter;
        private Texture2D      _readbackTex;

        public bool   IsRecording     => _isRecording;
        public string CurrentClipName => _currentClip?.name;

        public RecordStep(FaceRigRecording recording, int targetFps, bool captureReference, int referenceSize)
        {
            _recording        = recording;
            _minInterval      = 1f / targetFps;
            _captureReference = captureReference;
            _referenceSize    = referenceSize;
        }

        public void StartRecording(string clipName)
        {
            // Unique name: clip_00, clip_01, clip_02...
            string uniqueName = clipName;
            int suffix = 0;
            while (_recording.GetClip(uniqueName) != null)
            {
                uniqueName = $"{clipName}_{suffix:D2}";
                suffix++;
            }

            _currentClip = _recording.CreateClip(uniqueName);
            _recordStartTime = Time.time;
            _lastSnapshotTime = -1f;
            _frameCounter = 0;
            _isRecording = true;

            if (_captureReference)
            {
                _framesDir = Path.Combine(Application.dataPath, "Recordings", uniqueName);
                Directory.CreateDirectory(_framesDir);
                _currentClip.referenceFramesPath = $"Assets/Recordings/{uniqueName}";
            }
        }

        public void StopRecording()
        {
            _isRecording = false;
            _currentClip = null;
            _framesDir = null;

            if (_readbackTex != null)
            {
                Object.Destroy(_readbackTex);
                _readbackTex = null;
            }
        }

        public void Execute(FaceRigContext ctx)
        {
            if (!_isRecording || _currentClip == null) return;

            float elapsed = Time.time - _recordStartTime;

            if (elapsed - _lastSnapshotTime < _minInterval) return;
            _lastSnapshotTime = elapsed;

            var snapshot = new FrameSnapshot
            {
                timestamp = elapsed,

                activations = ctx.Activations
                    .Select(kv => new TagFloat { tag = kv.Key, value = kv.Value })
                    .ToArray(),

                strains = ctx.Strains
                    .Select(kv => new TagFloat { tag = kv.Key, value = kv.Value })
                    .ToArray(),

                irisPositions = ctx.IrisPositions
                    .Select(kv => new TagVector2 { tag = kv.Key, value = kv.Value })
                    .ToArray()
            };

            _currentClip.frames.Add(snapshot);

            if (_captureReference)
                CaptureReferenceFrame();
        }

        private void CaptureReferenceFrame()
        {
            var imageSource = ImageSourceProvider.ImageSource;
            if (imageSource == null) return;

            var tex = imageSource.GetCurrentTexture();
            if (tex == null) return;

            // Downscale to _referenceSize and readback
            float aspect = (float)tex.height / tex.width;
            int w = _referenceSize;
            int h = Mathf.RoundToInt(_referenceSize * aspect);

            var rt = RenderTexture.GetTemporary(w, h, 0);
            Graphics.Blit(tex, rt);

            var prev = RenderTexture.active;
            RenderTexture.active = rt;

            if (_readbackTex == null || _readbackTex.width != w || _readbackTex.height != h)
            {
                if (_readbackTex != null) Object.Destroy(_readbackTex);
                _readbackTex = new Texture2D(w, h, TextureFormat.RGB24, false);
            }

            _readbackTex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            _readbackTex.Apply();

            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);

            // Encode and save
            var jpg = _readbackTex.EncodeToJPG(75);
            var path = Path.Combine(_framesDir, $"frame_{_frameCounter:D4}.jpg");
            File.WriteAllBytes(path, jpg);
            _frameCounter++;
        }
    }
}
