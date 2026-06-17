namespace FaceRig.Core
{
    public interface IPipelineStep
    {
        void Execute(FaceRigContext ctx);
    }
}