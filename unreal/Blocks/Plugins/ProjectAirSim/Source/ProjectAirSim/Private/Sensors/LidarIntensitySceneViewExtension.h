#pragma once

#include "RHI.h"
#include "RHIGPUReadback.h"
#include "RHIResources.h"
#include "SceneViewExtension.h"

#include "LidarPointCloudCS.h"

class FLidarIntensitySceneViewExtension : public FSceneViewExtensionBase {
 public:
  FLidarIntensitySceneViewExtension(const FAutoRegister& AutoRegister,
                                    TWeakObjectPtr<UTextureRenderTarget2D> InRenderTarget2D);
  virtual ~FLidarIntensitySceneViewExtension();

  //~ Begin FSceneViewExtensionBase Interface
  virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override {};
  virtual void SetupView(FSceneViewFamily& InViewFamily,
                         FSceneView& InView) override {};
  virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override {};
  virtual void PreRenderViewFamily_RenderThread(
      FRHICommandListImmediate& RHICmdList,
      FSceneViewFamily& InViewFamily) override {};
  virtual void PreRenderView_RenderThread(FRHICommandListImmediate& RHICmdList,
                                          FSceneView& InView) override {};
  virtual void PostRenderBasePass_RenderThread(
      FRHICommandListImmediate& RHICmdList, FSceneView& InView) override {};

  // Only implement this, called right before post processing begins.
  virtual void PrePostProcessPass_RenderThread(
      FRDGBuilder& GraphBuilder, const FSceneView& View,
      const FPostProcessingInputs& Inputs) override;
  virtual bool IsActiveThisFrame_Internal(
      const FSceneViewExtensionContext& Context) const override;

  //~ End FSceneViewExtensionBase Interface

  bool IsValidForBoundRenderTarget(const FSceneViewFamily& Family) const;
  void UpdateParameters(FLidarPointCloudCSParameters& params);

  // Called from the game thread (after FlushRenderingCommands) to copy the
  // previous frame's GPU compute result into LidarPointCloudData.
  void CopyReadback();

  BEGIN_SHADER_PARAMETER_STRUCT(FCopyBufferToCPUPass, )
    RDG_BUFFER_ACCESS(Buffer, ERHIAccess::CopySrc)
  END_SHADER_PARAMETER_STRUCT()

public:
  std::vector<FVector4> LidarPointCloudData;

private:
  std::queue<FLidarPointCloudCSParameters> CSParamsQ;
  TWeakObjectPtr<UTextureRenderTarget2D> RenderTarget2D;

  FRHIGPUBufferReadback* PointCloudReadback = nullptr;
  uint32 PendingReadbackSize = 0;
};