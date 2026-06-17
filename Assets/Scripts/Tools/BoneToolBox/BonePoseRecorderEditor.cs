#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(BonePoseRecorder))]
public class BonePoseRecorderEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var r = (BonePoseRecorder)target;

        GUILayout.Space(10);
        EditorGUILayout.LabelField("Tools", EditorStyles.boldLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Auto Collect"))
            {
                Undo.RecordObject(r, "Auto Collect Bones");
                r.AutoCollectBones();
                EditorUtility.SetDirty(r);
            }

            if (GUILayout.Button("Clear List"))
            {
                Undo.RecordObject(r, "Clear Bones List");
                r.bones.Clear();
                EditorUtility.SetDirty(r);
            }
        }

        GUILayout.Space(6);

        using (new EditorGUILayout.HorizontalScope())
        {
            GUI.enabled = r.poseAsset != null;
            if (GUILayout.Button("Save Default Pose"))
            {
                r.SaveDefaultPose();
            }

            if (GUILayout.Button("Restore Pose"))
            {
                r.RestoreDefaultPose();
            }
            GUI.enabled = true;
        }

        if (r.poseAsset == null)
        {
            EditorGUILayout.HelpBox("Assign Pose Asset to enable Save/Restore.", MessageType.Warning);
        }
    }
}
#endif