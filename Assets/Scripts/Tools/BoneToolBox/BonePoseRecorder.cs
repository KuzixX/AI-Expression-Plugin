using System.Collections.Generic;
using UnityEngine;

public class BonePoseRecorder : MonoBehaviour
{
    [Header("Setup")]
    public Transform root;                 // Root for relative paths. If null -> this.transform
    public BoneDefaultPose poseAsset;

    [Header("Collect")]
    public bool includeRoot = false;
    public bool includeInactive = true;

    [Tooltip("Bones on these layers (and their children) will be excluded during AutoCollect.")]
    public LayerMask excludeLayers;

    public List<Transform> bones = new List<Transform>();

    public Transform EffectiveRoot => root != null ? root : transform;

    // ---------- Public API ----------

    public void AutoCollectBones()
    {
        bones.Clear();
        var r = EffectiveRoot;
        if (r == null) return;

        CollectRecursive(r, bones, includeInactive, excludeLayers);

        if (!includeRoot)
            bones.Remove(r);

        // Remove duplicates
        var set = new HashSet<Transform>();
        for (int i = bones.Count - 1; i >= 0; i--)
        {
            if (!set.Add(bones[i])) bones.RemoveAt(i);
        }
    }

    public void SaveDefaultPose()
    {
        if (poseAsset == null)
        {
            Debug.LogError("[BonePoseRecorder] Pose Asset is null");
            return;
        }

        var r = EffectiveRoot;
        if (r == null)
        {
            Debug.LogError("[BonePoseRecorder] Root is null");
            return;
        }

        poseAsset.bones.Clear();

        for (int i = 0; i < bones.Count; i++)
        {
            var b = bones[i];
            if (b == null) continue;

            var data = new BoneDefaultPose.BoneData
            {
                path = GetPathRelativeToRoot(r, b),
                localPosition = b.localPosition,
                localRotation = b.localRotation,
                localScale = b.localScale
            };

            poseAsset.bones.Add(data);
        }

#if UNITY_EDITOR
        UnityEditor.EditorUtility.SetDirty(poseAsset);
        UnityEditor.AssetDatabase.SaveAssets();
#endif

        Debug.Log($"[BonePoseRecorder] Saved default pose: {poseAsset.bones.Count} bones.");
    }

    public void RestoreDefaultPose()
    {
        if (poseAsset == null)
        {
            Debug.LogError("[BonePoseRecorder] Pose Asset is null");
            return;
        }

        var r = EffectiveRoot;
        if (r == null)
        {
            Debug.LogError("[BonePoseRecorder] Root is null");
            return;
        }

        int applied = 0;
        int missing = 0;

        foreach (var data in poseAsset.bones)
        {
            if (string.IsNullOrEmpty(data.path))
            {
                missing++;
                continue;
            }

            var bone = r.Find(data.path);
            if (bone == null)
            {
                missing++;
                continue;
            }

#if UNITY_EDITOR
            UnityEditor.Undo.RecordObject(bone, "Restore Bone Pose");
#endif
            bone.localPosition = data.localPosition;
            bone.localRotation = data.localRotation;
            bone.localScale = data.localScale;
            applied++;
        }

        Debug.Log($"[BonePoseRecorder] Restored pose. Applied: {applied}, Missing: {missing}");
    }

    // ---------- Helpers ----------

    private static void CollectRecursive(
        Transform t,
        List<Transform> list,
        bool includeInactive,
        LayerMask excludeLayers)
    {
        if (!includeInactive && !t.gameObject.activeInHierarchy)
            return;

        // If this transform is on excluded layer -> skip it AND its whole subtree
        if (IsInLayerMask(t.gameObject.layer, excludeLayers))
            return;

        list.Add(t);

        for (int i = 0; i < t.childCount; i++)
        {
            CollectRecursive(t.GetChild(i), list, includeInactive, excludeLayers);
        }
    }

    private static bool IsInLayerMask(int layer, LayerMask mask)
    {
        return (mask.value & (1 << layer)) != 0;
    }

    private static string GetPathRelativeToRoot(Transform root, Transform target)
    {
        if (target == root) return string.Empty;

        var stack = new Stack<string>();
        var cur = target;

        while (cur != null && cur != root)
        {
            stack.Push(cur.name);
            cur = cur.parent;
        }

        if (cur != root) return string.Empty;

        return string.Join("/", stack.ToArray());
    }
}