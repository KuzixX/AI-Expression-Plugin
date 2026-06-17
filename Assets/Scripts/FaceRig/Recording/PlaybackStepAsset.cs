using System.Collections.Generic;
using FaceRig.Core;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Playback")]
    public class PlaybackStepAsset : PipelineStepAsset
    {
        [SerializeField] private FaceRigRecording _recording;
        [SerializeField] private bool _loop = true;

        public override IPipelineStep CreateStep() => new PlaybackStep(_recording, _loop);
    }

    public class PlaybackStep : IPipelineStep
    {
        private readonly FaceRigRecording _recording;
        private readonly bool             _loop;

        private bool           _isPlaying;
        private float          _playbackStartTime;
        private RecordingClip  _currentClip;

        public bool   IsPlaying       => _isPlaying;
        public string CurrentClipName => _currentClip?.name;

        public PlaybackStep(FaceRigRecording recording, bool loop)
        {
            _recording = recording;
            _loop      = loop;
        }

        public void StartPlayback(string clipName)
        {
            if (_recording == null) return;
            _currentClip = _recording.GetClip(clipName);
            if (_currentClip == null || _currentClip.frames.Count == 0) return;

            _playbackStartTime = Time.time;
            _isPlaying = true;
        }

        public void StopPlayback()
        {
            _isPlaying = false;
            _currentClip = null;
        }

        public List<string> GetAvailableClipNames()
        {
            var names = new List<string>();
            if (_recording == null) return names;
            for (int i = 0; i < _recording.clips.Count; i++)
                names.Add(_recording.clips[i].name);
            return names;
        }

        public void Execute(FaceRigContext ctx)
        {
            if (!_isPlaying || _currentClip == null || _currentClip.frames.Count == 0)
                return;

            float elapsed = Time.time - _playbackStartTime;
            var frames = _currentClip.frames;
            float duration = frames[frames.Count - 1].timestamp;

            if (duration <= 0f)
            {
                _isPlaying = false;
                return;
            }

            if (_loop)
            {
                elapsed %= duration;
            }
            else if (elapsed > duration)
            {
                _isPlaying = false;
                return;
            }

            int frameIdx = FindFrameIndex(frames, elapsed);
            var snapshot = frames[frameIdx];

            for (int i = 0; i < snapshot.activations.Length; i++)
                ctx.Activations[snapshot.activations[i].tag] = snapshot.activations[i].value;

            for (int i = 0; i < snapshot.strains.Length; i++)
                ctx.Strains[snapshot.strains[i].tag] = snapshot.strains[i].value;

            for (int i = 0; i < snapshot.irisPositions.Length; i++)
                ctx.IrisPositions[snapshot.irisPositions[i].tag] = snapshot.irisPositions[i].value;
        }

        private int FindFrameIndex(List<FrameSnapshot> frames, float time)
        {
            int lo = 0, hi = frames.Count - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (frames[mid].timestamp < time)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            return lo;
        }
    }
}
