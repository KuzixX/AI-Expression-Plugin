using System;
using System.Collections.Generic;
using FaceRig.Data;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [Serializable]
    public struct TagFloat
    {
        public FaceMuscleAnchorTag tag;
        public float value;
    }

    [Serializable]
    public struct TagVector2
    {
        public FaceLandMarkTag tag;
        public Vector2 value;
    }

    [Serializable]
    public struct FrameSnapshot
    {
        public float timestamp;
        public TagFloat[] activations;
        public TagFloat[] strains;
        public TagVector2[] irisPositions;
    }

    [Serializable]
    public class RecordingClip
    {
        public string name;
        public string referenceFramesPath;
        public List<FrameSnapshot> frames = new();
    }

    [CreateAssetMenu(menuName = "FaceRig/Recording")]
    public class FaceRigRecording : ScriptableObject
    {
        public List<RecordingClip> clips = new();

        public RecordingClip CreateClip(string clipName)
        {
            var clip = new RecordingClip { name = clipName };
            clips.Add(clip);
            return clip;
        }

        public RecordingClip GetClip(string clipName)
        {
            for (int i = 0; i < clips.Count; i++)
                if (clips[i].name == clipName)
                    return clips[i];
            return null;
        }

        public void RemoveClip(string clipName)
        {
            for (int i = clips.Count - 1; i >= 0; i--)
                if (clips[i].name == clipName)
                    clips.RemoveAt(i);
        }

        public void Clear()
        {
            clips.Clear();
        }
    }
}
