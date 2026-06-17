using System;
using System.Collections.Generic;
using UnityEngine;

namespace FaceRig.Data
{
   [Serializable]
   public class FaceLandmarkDescription
   {
      [SerializeField] public FaceLandMarkTag faceLandMarkTag;
      [SerializeField] public int idx;
      [SerializeField] public List<FaceLandmarkTargetDescription> targets = new();
   }
}