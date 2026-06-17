using System;
using System.Collections.Generic;
using FaceRig.Core;
using FaceRig.Data;
using Unity.InferenceEngine;
using UnityEngine;

namespace FaceMuscle.FaceRigPipeline.Core
{
    [CreateAssetMenu(menuName = "FaceRig/Steps/Denoiser (Neural)")]
    public class DenoiserStepAsset : PipelineStepAsset
    {
        [SerializeField] private ModelAsset  _modelAsset;
        [SerializeField] private TextAsset   _columnsJson;
        [SerializeField] private TextAsset   _normStatsJson;
        [SerializeField] private BackendType _backend = BackendType.CPU;
        [SerializeField] [Range(0f, 1f)] private float _strength = 1f;
        [SerializeField] private bool _debugLog;

        public override IPipelineStep CreateStep()
            => new DenoiserStep(_modelAsset, _columnsJson, _normStatsJson, _backend, _strength, _debugLog);
    }

    public class DenoiserStep : IPipelineStep, IDisposable
    {
        private readonly Worker        _worker;
        private readonly ColumnMapping _mapping;
        private readonly int           _inputDim;
        private readonly float         _strength;
        private readonly bool          _debugLog;
        private int _debugFrameCounter;

        public DenoiserStep(ModelAsset modelAsset, TextAsset columnsJson,
                            TextAsset normStatsJson, BackendType backend, float strength, bool debugLog)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, backend);

            _mapping = new ColumnMapping(columnsJson.text, normStatsJson.text);
            _inputDim = _mapping.Count;
            _strength = Mathf.Clamp01(strength);
            _debugLog = debugLog;

            if (_debugLog)
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("[Denoiser] Norm stats loaded:");
                var names = _mapping.ColumnNames;
                var mins = _mapping.ColMin;
                var ranges = _mapping.ColRange;
                for (int i = 0; i < _inputDim; i++)
                    sb.AppendLine($"  {names[i],-45} min={mins[i],12:F8}  range={ranges[i],12:F8}");
                Debug.Log(sb.ToString());
            }
        }

        public void Execute(FaceRigContext ctx)
        {
            if (_strength < 0.001f) return;
            if (ctx.Activations.Count == 0 && ctx.Strains.Count == 0) return;

            // Pack context → normalized float[]
            var input = new float[_inputDim];
            _mapping.PackFromContext(ctx, input);

            // Save raw denormalized for debug
            float[] rawDenorm = null;
            if (_debugLog)
            {
                rawDenorm = new float[_inputDim];
                _mapping.Denormalize(input, rawDenorm);
            }

            // Run inference
            using var inputTensor = new Tensor<float>(new TensorShape(1, _inputDim), input);
            _worker.SetInput("raw_input", inputTensor);
            _worker.Schedule();

            var outputTensor = _worker.PeekOutput("cleaned_output") as Tensor<float>;
            outputTensor.CompleteAllPendingOperations();
            using var cpuOutput = outputTensor.ReadbackAndClone() as Tensor<float>;

            // Blend: lerp(raw_normalized, denoised_normalized, strength)
            var output = new float[_inputDim];
            for (int i = 0; i < _inputDim; i++)
                output[i] = input[i] + (cpuOutput[i] - input[i]) * _strength;

            // Debug log every 30 frames
            if (_debugLog && _debugFrameCounter++ % 30 == 0)
            {
                var cleanDenorm = new float[_inputDim];
                _mapping.Denormalize(output, cleanDenorm);

                var sb = new System.Text.StringBuilder();
                var names = _mapping.ColumnNames;

                // Dump normalized input + Sentis raw output for comparison with Python
                sb.AppendLine("[Denoiser] NORMALIZED input → Sentis output:");
                for (int i = 0; i < _inputDim; i++)
                    sb.AppendLine($"  {names[i],-45} in={input[i],10:F6}  sentis_out={cpuOutput[i],10:F6}");
                sb.AppendLine();

                sb.AppendLine("[Denoiser] DENORMALIZED Raw → Clean:");
                for (int i = 0; i < _inputDim; i++)
                {
                    float delta = cleanDenorm[i] - rawDenorm[i];
                    string flag = Mathf.Abs(delta) > 0.02f ? " <<<" : "";
                    sb.AppendLine($"  {names[i],-45} raw={rawDenorm[i],9:F5}  clean={cleanDenorm[i],9:F5}  Δ={delta,+9:F5}{flag}");
                }
                Debug.Log(sb.ToString());
            }

            _mapping.UnpackToContext(ctx, output);
        }

        public void Dispose()
        {
            _worker?.Dispose();
        }
    }

    /// <summary>
    /// Maps between FaceRigContext and flat float[] with per-column min-max normalization.
    /// </summary>
    public class ColumnMapping
    {
        public int Count => _entries.Length;
        public string[] ColumnNames => _columnNames;
        public float[] ColMin => _colMin;
        public float[] ColRange => _colRange;

        private readonly Entry[] _entries;
        private readonly string[] _columnNames;
        private readonly float[] _colMin;
        private readonly float[] _colRange;

        private enum ColumnType { Activation, Strain, IrisX, IrisY }

        private struct Entry
        {
            public ColumnType type;
            public FaceMuscleAnchorTag muscleTag;
            public FaceLandMarkTag landmarkTag;
        }

        public ColumnMapping(string columnsJsonText, string normStatsJsonText)
        {
            var columns = ParseJsonArray(columnsJsonText);
            var normStats = ParseNormStats(normStatsJsonText);

            _entries     = new Entry[columns.Count];
            _columnNames = new string[columns.Count];
            _colMin      = new float[columns.Count];
            _colRange    = new float[columns.Count];

            for (int i = 0; i < columns.Count; i++)
            {
                var col = columns[i];
                _columnNames[i] = col;

                // Norm stats
                if (normStats.TryGetValue(col, out var stats))
                {
                    _colMin[i]   = stats.min;
                    _colRange[i] = stats.max - stats.min;
                    if (_colRange[i] < 1e-8f) _colRange[i] = 1f;
                }
                else
                {
                    _colMin[i]   = 0f;
                    _colRange[i] = 1f;
                }

                // Column type
                if (col.StartsWith("act_"))
                {
                    _entries[i] = new Entry
                    {
                        type = ColumnType.Activation,
                        muscleTag = Enum.Parse<FaceMuscleAnchorTag>(col.Substring(4)),
                    };
                }
                else if (col.StartsWith("str_"))
                {
                    _entries[i] = new Entry
                    {
                        type = ColumnType.Strain,
                        muscleTag = Enum.Parse<FaceMuscleAnchorTag>(col.Substring(4)),
                    };
                }
                else if (col.StartsWith("iris_") && col.EndsWith("_x"))
                {
                    _entries[i] = new Entry
                    {
                        type = ColumnType.IrisX,
                        landmarkTag = Enum.Parse<FaceLandMarkTag>(col.Substring(5, col.Length - 7)),
                    };
                }
                else if (col.StartsWith("iris_") && col.EndsWith("_y"))
                {
                    _entries[i] = new Entry
                    {
                        type = ColumnType.IrisY,
                        landmarkTag = Enum.Parse<FaceLandMarkTag>(col.Substring(5, col.Length - 7)),
                    };
                }
            }
        }

        /// <summary>Pack context → normalized [-1..1] for network input.</summary>
        public void PackFromContext(FaceRigContext ctx, float[] dst)
        {
            for (int i = 0; i < _entries.Length; i++)
            {
                ref var e = ref _entries[i];
                float val = 0f;

                switch (e.type)
                {
                    case ColumnType.Activation:
                        ctx.Activations.TryGetValue(e.muscleTag, out val);
                        break;
                    case ColumnType.Strain:
                        ctx.Strains.TryGetValue(e.muscleTag, out val);
                        break;
                    case ColumnType.IrisX:
                        if (ctx.IrisPositions.TryGetValue(e.landmarkTag, out var ix)) val = ix.x;
                        break;
                    case ColumnType.IrisY:
                        if (ctx.IrisPositions.TryGetValue(e.landmarkTag, out var iy)) val = iy.y;
                        break;
                }

                // min-max normalize to [-1..1]
                dst[i] = (val - _colMin[i]) / _colRange[i] * 2f - 1f;
            }
        }

        /// <summary>Convert normalized [-1..1] array back to real values (for debug).</summary>
        public void Denormalize(float[] normed, float[] dst)
        {
            for (int i = 0; i < _entries.Length; i++)
                dst[i] = (normed[i] + 1f) / 2f * _colRange[i] + _colMin[i];
        }

        /// <summary>Unpack network output [-1..1] → denormalized context.
        /// Iris channels are skipped (passed through raw) because the network
        /// reconstructs them poorly.</summary>
        public void UnpackToContext(FaceRigContext ctx, float[] src)
        {
            for (int i = 0; i < _entries.Length; i++)
            {
                ref var e = ref _entries[i];

                // min-max denormalize from [-1..1]
                float val = (src[i] + 1f) / 2f * _colRange[i] + _colMin[i];

                switch (e.type)
                {
                    case ColumnType.Activation:
                        ctx.Activations[e.muscleTag] = Mathf.Clamp01(val);
                        break;
                    case ColumnType.Strain:
                        ctx.Strains[e.muscleTag] = val;
                        break;
                    // Iris — skip, keep raw values from context
                    case ColumnType.IrisX:
                    case ColumnType.IrisY:
                        break;
                }
            }
        }

        private static List<string> ParseJsonArray(string json)
        {
            var result = new List<string>();
            int i = json.IndexOf('[');
            if (i < 0) return result;
            while (true)
            {
                int start = json.IndexOf('"', i + 1);
                if (start < 0) break;
                int end = json.IndexOf('"', start + 1);
                if (end < 0) break;
                result.Add(json.Substring(start + 1, end - start - 1));
                i = end;
            }
            return result;
        }

        private struct MinMax { public float min, max; }

        /// <summary>
        /// Parses {"col_name": {"min": X, "max": Y}, ...} by tracking brace nesting.
        /// </summary>
        private static Dictionary<string, MinMax> ParseNormStats(string json)
        {
            var result = new Dictionary<string, MinMax>();

            // Find outer opening brace
            int pos = json.IndexOf('{');
            if (pos < 0) return result;
            pos++; // skip '{'

            while (pos < json.Length)
            {
                // Find column name key
                int keyStart = json.IndexOf('"', pos);
                if (keyStart < 0) break;
                int keyEnd = json.IndexOf('"', keyStart + 1);
                if (keyEnd < 0) break;
                string key = json.Substring(keyStart + 1, keyEnd - keyStart - 1);
                pos = keyEnd + 1;

                // Find the opening '{' of this column's value object
                int objStart = json.IndexOf('{', pos);
                if (objStart < 0) break;

                // Find matching closing '}'
                int objEnd = json.IndexOf('}', objStart + 1);
                if (objEnd < 0) break;

                // Parse the inner object: {"min": X, "max": Y}
                string inner = json.Substring(objStart + 1, objEnd - objStart - 1);

                float minVal = 0f, maxVal = 0f;

                int minIdx = inner.IndexOf("\"min\"");
                if (minIdx >= 0)
                {
                    int colon = inner.IndexOf(':', minIdx + 5);
                    if (colon >= 0)
                    {
                        int end = inner.IndexOfAny(new[] { ',', '}' }, colon + 1);
                        if (end < 0) end = inner.Length;
                        float.TryParse(inner.Substring(colon + 1, end - colon - 1).Trim(),
                            System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out minVal);
                    }
                }

                int maxIdx = inner.IndexOf("\"max\"");
                if (maxIdx >= 0)
                {
                    int colon = inner.IndexOf(':', maxIdx + 5);
                    if (colon >= 0)
                    {
                        int end = inner.IndexOfAny(new[] { ',', '}' }, colon + 1);
                        if (end < 0) end = inner.Length;
                        float.TryParse(inner.Substring(colon + 1, end - colon - 1).Trim(),
                            System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out maxVal);
                    }
                }

                result[key] = new MinMax { min = minVal, max = maxVal };
                pos = objEnd + 1;
            }

            return result;
        }
    }
}
