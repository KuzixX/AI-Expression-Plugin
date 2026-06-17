using System.Collections;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using UnityEngine;
using UnityEngine.Rendering;

namespace Mediapipe.Unity.Sample.FaceLandmarkDetection
{
  public class FaceLandmarkerRunner : VisionTaskApiRunner<FaceLandmarker>
  {
    [SerializeField] private FaceLandmarkerResultAnnotationController faceLandmarkerResultAnnotationController;

    private Experimental.TextureFramePool _textureFramePool;

    public readonly FaceLandmarkDetectionConfig config = new FaceLandmarkDetectionConfig();

    // Храним последний результат
    private readonly object _resultLock = new object();
    private FaceLandmarkerResult _latestResult;

    public override void Stop()
    {
      base.Stop();
      _textureFramePool?.Dispose();
      _textureFramePool = null;
    }

    /// <summary>
    /// Возвращает последнюю сохраненную копию результата.
    /// Может вернуть default, если лицо еще не было найдено.
    /// </summary>
    public FaceLandmarkerResult GetLatestResult()
    {
      lock (_resultLock)
      {
        return _latestResult;
      }
    }

    /// <summary>
    /// Возвращает копию последнего результата в out-параметр.
    /// Более безопасный вариант, если не хочешь отдавать внутреннюю ссылку.
    /// </summary>
    public bool TryGetLatestResult(ref FaceLandmarkerResult result)
    {
      lock (_resultLock)
      {
        if (_latestResult.faceLandmarks == null)
          return false;

        _latestResult.CloneTo(ref result);
        return true;
      }
    }

    private void SaveLatestResult(FaceLandmarkerResult result)
    {
      lock (_resultLock)
      {
        result.CloneTo(ref _latestResult);
      }
    }

    protected override IEnumerator Run()
    {
      yield return AssetLoader.PrepareAssetAsync(config.ModelPath);

      var options = config.GetFaceLandmarkerOptions(
        config.RunningMode == Tasks.Vision.Core.RunningMode.LIVE_STREAM
          ? OnFaceLandmarkDetectionOutput
          : null);

      taskApi = FaceLandmarker.CreateFromOptions(options, GpuManager.GpuResources);
      var imageSource = ImageSourceProvider.ImageSource;

      yield return imageSource.Play();

      if (!imageSource.isPrepared)
      {
        Debug.LogError("Failed to start ImageSource, exiting...");
        yield break;
      }

      _textureFramePool = new Experimental.TextureFramePool(
        imageSource.textureWidth,
        imageSource.textureHeight,
        TextureFormat.RGBA32,
        10);

      screen.Initialize(imageSource);
      SetupAnnotationController(faceLandmarkerResultAnnotationController, imageSource);

      var transformationOptions = imageSource.GetTransformationOptions();
      var flipHorizontally = transformationOptions.flipHorizontally;
      var flipVertically = transformationOptions.flipVertically;
      var imageProcessingOptions =
        new Tasks.Vision.Core.ImageProcessingOptions(
          rotationDegrees: (int)transformationOptions.rotationAngle);

      AsyncGPUReadbackRequest req = default;
      var waitUntilReqDone = new WaitUntil(() => req.done);
      var waitForEndOfFrame = new WaitForEndOfFrame();
      var result = FaceLandmarkerResult.Alloc(options.numFaces);

      var canUseGpuImage =
        SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3 &&
        GpuManager.GpuResources != null;

      using var glContext = canUseGpuImage ? GpuManager.GetGlContext() : null;

      while (true)
      {
        if (isPaused)
        {
          yield return new WaitWhile(() => isPaused);
        }

        if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
        {
          yield return null;
          continue;
        }

        Image image;
        switch (config.ImageReadMode)
        {
          case ImageReadMode.GPU:
            if (!canUseGpuImage)
            {
              throw new System.Exception("ImageReadMode.GPU is not supported");
            }

            textureFrame.ReadTextureOnGPU(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
            image = textureFrame.BuildGPUImage(glContext);
            yield return waitForEndOfFrame;
            break;

          case ImageReadMode.CPU:
            yield return waitForEndOfFrame;
            textureFrame.ReadTextureOnCPU(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
            image = textureFrame.BuildCPUImage();
            textureFrame.Release();
            break;

          case ImageReadMode.CPUAsync:
          default:
            req = textureFrame.ReadTextureAsync(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
            yield return waitUntilReqDone;

            if (req.hasError)
            {
              Debug.LogWarning("Failed to read texture from the image source");
              continue;
            }

            image = textureFrame.BuildCPUImage();
            textureFrame.Release();
            break;
        }

        switch (taskApi.runningMode)
        {
          case Tasks.Vision.Core.RunningMode.IMAGE:
            if (taskApi.TryDetect(image, imageProcessingOptions, ref result))
            {
              SaveLatestResult(result);
              faceLandmarkerResultAnnotationController.DrawNow(result);
            }
            else
            {
              faceLandmarkerResultAnnotationController.DrawNow(default);
            }
            break;

          case Tasks.Vision.Core.RunningMode.VIDEO:
            if (taskApi.TryDetectForVideo(image, GetCurrentTimestampMillisec(), imageProcessingOptions, ref result))
            {
              SaveLatestResult(result);
              faceLandmarkerResultAnnotationController.DrawNow(result);
            }
            else
            {
              faceLandmarkerResultAnnotationController.DrawNow(default);
            }
            break;

          case Tasks.Vision.Core.RunningMode.LIVE_STREAM:
            taskApi.DetectAsync(image, GetCurrentTimestampMillisec(), imageProcessingOptions);
            break;
        }
      }
    }

    private void OnFaceLandmarkDetectionOutput(FaceLandmarkerResult result, Image image, long timestamp)
    {
      SaveLatestResult(result);
      faceLandmarkerResultAnnotationController.DrawLater(result);
    }
  }
}