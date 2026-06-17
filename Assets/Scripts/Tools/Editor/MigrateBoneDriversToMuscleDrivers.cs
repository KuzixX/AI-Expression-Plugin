using FaceMuscle.FaceRigPipeline.Drivers;
using FaceMuscle.Runtime.Systems;
using UnityEditor;
using UnityEngine;

public static class MigrateBoneDriversToMuscleDrivers
{
    [MenuItem("Tools/Migrate SBonePositionDriver → MuscleDriver")]
    public static void Migrate()
    {
        var drivers = Object.FindObjectsByType<SBonePositionDriver>(FindObjectsSortMode.None);

        int migrated = 0;

        foreach (var old in drivers)
        {
            var go = old.gameObject;

            // Пропускаем если MuscleDriver уже есть
            if (go.TryGetComponent<MuscleDriver>(out _))
            {
                Debug.Log($"[Migration] Skipped {go.name} — MuscleDriver already exists");
                continue;
            }

            // Добавляем MuscleDriver
            var muscle = go.AddComponent<MuscleDriver>();

            // Копируем данные через SerializedObject
            var soOld = new SerializedObject(old);
            var soNew = new SerializedObject(muscle);

            CopyProperty(soOld, "targets",          soNew, "targets");
            CopyProperty(soOld, "_currentPosition",  soNew, "currentPosition");
            CopyProperty(soOld, "_neutralPosition",  soNew, "neutralPosition");

            soNew.ApplyModifiedProperties();

            // Выключаем старый драйвер
            old.enabled = false;

            EditorUtility.SetDirty(go);
            migrated++;

            Debug.Log($"[Migration] Migrated {go.name}");
        }

        Debug.Log($"[Migration] Done. Migrated {migrated} drivers, total found {drivers.Length}");
    }

    private static void CopyProperty(SerializedObject src, string srcName, SerializedObject dst, string dstName)
    {
        var srcProp = src.FindProperty(srcName);
        var dstProp = dst.FindProperty(dstName);

        if (srcProp == null)
        {
            Debug.LogWarning($"[Migration] Source property '{srcName}' not found");
            return;
        }
        if (dstProp == null)
        {
            Debug.LogWarning($"[Migration] Dest property '{dstName}' not found");
            return;
        }

        dstProp.isExpanded = srcProp.isExpanded;

        // Для массивов — копируем поэлементно
        if (srcProp.isArray)
        {
            dstProp.arraySize = srcProp.arraySize;
            for (int i = 0; i < srcProp.arraySize; i++)
            {
                CopyPropertyValue(srcProp.GetArrayElementAtIndex(i), dstProp.GetArrayElementAtIndex(i));
            }
        }
        else
        {
            CopyPropertyValue(srcProp, dstProp);
        }
    }

    private static void CopyPropertyValue(SerializedProperty src, SerializedProperty dst)
    {
        switch (src.propertyType)
        {
            case SerializedPropertyType.ObjectReference:
                dst.objectReferenceValue = src.objectReferenceValue;
                break;
            case SerializedPropertyType.Integer:
                dst.intValue = src.intValue;
                break;
            case SerializedPropertyType.Float:
                dst.floatValue = src.floatValue;
                break;
            case SerializedPropertyType.String:
                dst.stringValue = src.stringValue;
                break;
            case SerializedPropertyType.Boolean:
                dst.boolValue = src.boolValue;
                break;
            case SerializedPropertyType.Enum:
                dst.enumValueIndex = src.enumValueIndex;
                break;
            case SerializedPropertyType.Vector3:
                dst.vector3Value = src.vector3Value;
                break;
            case SerializedPropertyType.Generic:
                // Для вложенных структур (JointTarget) — копируем дочерние свойства
                var srcIter = src.Copy();
                var dstIter = dst.Copy();
                var srcEnd = src.GetEndProperty();
                srcIter.NextVisible(true);
                dstIter.NextVisible(true);
                while (!SerializedProperty.EqualContents(srcIter, srcEnd))
                {
                    CopyPropertyValue(srcIter, dstIter);
                    if (!srcIter.NextVisible(false)) break;
                    dstIter.NextVisible(false);
                }
                break;
        }
    }
}
