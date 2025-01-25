//
// Created by magnus on 1/24/25.
//

#ifndef MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H
#define MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H

#include <torch/torch.h>
#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
// Wrap your raytracer in a Torch module
namespace VkRender::PathTracer {

    class PhotonRebuildModule : public torch::nn::Module {
    public:
        PhotonRebuildModule(PhotonTracer* rt);

        // forward() will trigger a ray trace and return an image tensor
        torch::Tensor forward(PhotonTracer::Settings& settings, std::shared_ptr<Scene> scene);

        float* getRenderedImage();

        torch::Tensor myParam;


        GPUData m_gpuData;

    private:
        PhotonTracer* m_photonRebuild;
    };

}


#endif //MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H
