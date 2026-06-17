#if UNITY_EDITOR
using UnityEditor;

namespace MusclesToolbox
{
    /// <summary>
    /// Optional interface for future extension. Not required, but keeps modules consistent.
    /// </summary>
    public interface IToolboxModule
    {
        void OnEnable(EditorWindow host);
        void OnDisable(EditorWindow host);

        void OnGUI(EditorWindow host);
        void OnSceneGUI(EditorWindow host, SceneView sceneView);
    }
}
#endif