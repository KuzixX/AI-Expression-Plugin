using System.Linq;
using FaceMuscle.FaceRigPipeline.Core;
using Mediapipe.Unity.Sample.FaceLandmarkDetection;
using UnityEngine;
using UnityEngine.InputSystem;

namespace FaceRig.Core
{
    public class StartUp : MonoBehaviour
    {
        [SerializeField] private FaceRigPipeline       _pipeline;
        [SerializeField] private FaceLandmarkerRunner  _landmarkerRunner;
        [SerializeField] private Key                   _recordKey = Key.R;
        [SerializeField] private string                _nextClipName = "clip";
        private IPipelineStep[]                        _runtimeSteps;
        private FaceRigContext                         _ctx;
        private RecordStep                             _recordStep;
        private int                                    _clipCounter;

        private void Start()
        {
            _ctx = new FaceRigContext { LandmarkerRunner = _landmarkerRunner };

            // Шаги из ассета (солверы, фильтры и т.д.)
            var assetSteps = _pipeline.steps.Select(s => s.CreateStep()).ToList();

            // Отделяем RecordStep — он должен идти последним, после всех драйверов
            var recordSteps = assetSteps.Where(s => s is RecordStep).ToList();
            var otherAssetSteps = assetSteps.Where(s => s is not RecordStep);

            // Драйверы со сцены (MonoBehaviour, реализуют IPipelineStep)
            var sceneDrivers = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None)
                .OfType<IPipelineStep>();

            // Порядок: SO-солверы → сцена-драйверы → RecordStep (последний, снимает все данные)
            _runtimeSteps = otherAssetSteps.Concat(sceneDrivers).Concat(recordSteps).ToArray();

            // Кешируем RecordStep если есть в пайплайне
            foreach (var step in _runtimeSteps)
            {
                if (step is RecordStep rs)
                {
                    _recordStep = rs;
                    break;
                }
            }
        }

        private void Update()
        {
            // Запись по горячей клавише
            if (_recordStep != null && Keyboard.current != null && Keyboard.current[_recordKey].wasPressedThisFrame)
            {
                if (_recordStep.IsRecording)
                {
                    Debug.Log($"[FaceRig] Recording stopped: \"{_recordStep.CurrentClipName}\"");
                    _recordStep.StopRecording();
                }
                else
                {
                    string clipName = $"{_nextClipName}_{_clipCounter++}";
                    _recordStep.StartRecording(clipName);
                    Debug.Log($"[FaceRig] Recording started: \"{clipName}\"");
                }
            }

            _ctx.Clear();
            foreach (var step in _runtimeSteps)
                step.Execute(_ctx);
        }
    }
}