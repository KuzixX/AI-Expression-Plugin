using System;
using UnityEngine;

namespace FaceRig.Data
{
    [Serializable]
    public class EyeAxisSpaceDescrtiption
    {
        [SerializeField] private int startIdx;
        [SerializeField] private int endIdx;

        public int StartIdx => startIdx;
        public int EndIdx   => endIdx;
    }
}