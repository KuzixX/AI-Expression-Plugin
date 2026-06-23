#if UNITY_EDITOR
using System.Collections.Generic;
using System.Text;
using FaceMuscle.FaceRigPipeline.Drivers;
using FaceRig.Data;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace FaceRig.EditorTools
{
    /// <summary>
    /// Возвращает потерянные MuscleDriver на джойнты и прописывает ссылки
    /// (currentPosition / neutralPosition / target) по конвенции имён рига:
    ///   currentPosition = сам джойнт
    ///   neutralPosition = "<joint>_Neutral_Position"
    ///   target          = "<joint>_TARGET"
    /// targetTag берётся из таблицы Entries.
    ///
    /// Использование: открой Male.prefab в Prefab Mode (двойной клик) ИЛИ выдели
    /// корень рига в сцене → Tools ▸ FaceRig ▸ Restore Missing Muscle Drivers →
    /// сохрани (Ctrl+S).
    /// </summary>
    public static class MuscleDriverRestorer
    {
        private struct Entry
        {
            public string Joint;
            public FaceMuscleAnchorTag Tag;
            public Entry(string joint, FaceMuscleAnchorTag tag) { Joint = joint; Tag = tag; }
        }

        // Таблица «джойнт → мышца». Расширяй при необходимости.
        private static readonly Entry[] Entries =
        {
            // верхняя губа — levator labii superioris (поднимает)
            new Entry("L_LIP_MIDDLE_UP",   FaceMuscleAnchorTag.LLevatorLabiiSuperiorisLandmark),
            new Entry("R_LIP_MIDDLE_UP",   FaceMuscleAnchorTag.RLevatorLabiiSuperiorisLandmark),
            // нижняя губа — depressor labii inferioris (опускает)
            new Entry("L_LIP_MIDDLE_DOWN", FaceMuscleAnchorTag.LDepressorLabiiInferiorisLandmark),
            new Entry("R_LIP_MIDDLE_DOWN", FaceMuscleAnchorTag.RDepressorLabiiInferiorisLandmark),
            // щёки — zygomaticus major (тянет щёку/уголок вверх, улыбка)
            new Entry("L_CHEEK", FaceMuscleAnchorTag.lZygomaticusMajorLandmark),
            new Entry("R_CHEEK", FaceMuscleAnchorTag.rZygomaticusMajorLandmark),
        };

        private const string NeutralSuffix = "_Neutral_Position";
        private const string TargetSuffix  = "_TARGET";

        [MenuItem("Tools/FaceRig/Restore Missing Muscle Drivers (Lips + Cheeks)")]
        private static void Restore()
        {
            GameObject root = ResolveRoot();
            if (root == null)
            {
                EditorUtility.DisplayDialog("Restore Muscle Drivers",
                    "Открой Male.prefab в Prefab Mode (двойной клик по префабу) " +
                    "или выдели корень рига в сцене, затем запусти снова.", "OK");
                return;
            }

            var map = new Dictionary<string, Transform>();
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                map[t.name] = t;   // имена джойнтов в риге уникальны

            int added = 0, wired = 0;
            var log = new StringBuilder();

            foreach (var e in Entries)
            {
                Transform joint = Find(map, e.Joint);
                if (joint == null) { log.AppendLine($"❌ нет джойнта '{e.Joint}' — пропуск"); continue; }

                Transform neutral = Find(map, e.Joint + NeutralSuffix);
                Transform target  = Find(map, e.Joint + TargetSuffix);

                var driver = joint.GetComponent<MuscleDriver>();
                if (driver == null)
                {
                    driver = Undo.AddComponent<MuscleDriver>(joint.gameObject);
                    added++;
                }

                var so = new SerializedObject(driver);
                so.FindProperty("currentPosition").objectReferenceValue = joint;
                if (neutral != null)
                    so.FindProperty("neutralPosition").objectReferenceValue = neutral;

                var targets = so.FindProperty("targets");
                targets.arraySize = 1;
                var el = targets.GetArrayElementAtIndex(0);
                el.FindPropertyRelative("targetTag").enumValueIndex = (int)e.Tag;
                el.FindPropertyRelative("target").objectReferenceValue = target;
                el.FindPropertyRelative("activation").floatValue = 0f;
                el.FindPropertyRelative("weight").floatValue = 1f;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(driver);
                wired++;

                log.AppendLine(
                    $"✅ {e.Joint}: tag={e.Tag}, " +
                    $"neutral={(neutral ? neutral.name : "— НЕ НАЙДЕН")}, " +
                    $"target={(target ? target.name : "— НЕ НАЙДЕН")}");
            }

            var stage = PrefabStageUtility.GetCurrentPrefabStage();
            if (stage != null) EditorSceneManager.MarkSceneDirty(stage.scene);
            else if (!Application.isPlaying) EditorSceneManager.MarkAllScenesDirty();

            string summary = $"MuscleDriver: добавлено {added}, прописано {wired}.\n\n{log}";
            Debug.Log("[MuscleDriverRestorer]\n" + summary);
            EditorUtility.DisplayDialog("Restore Muscle Drivers",
                summary + "\nНе забудь сохранить (Ctrl+S).", "OK");
        }

        private static GameObject ResolveRoot()
        {
            var stage = PrefabStageUtility.GetCurrentPrefabStage();
            if (stage != null) return stage.prefabContentsRoot;
            if (Selection.activeGameObject != null)
                return Selection.activeGameObject.transform.root.gameObject;
            return null;
        }

        private static Transform Find(Dictionary<string, Transform> map, string name)
        {
            if (map.TryGetValue(name, out var t)) return t;
            foreach (var kv in map)
                if (string.Equals(kv.Key, name, System.StringComparison.OrdinalIgnoreCase))
                    return kv.Value;
            return null;
        }
    }
}
#endif
