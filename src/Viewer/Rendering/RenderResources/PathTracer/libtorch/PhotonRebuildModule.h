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
        PhotonRebuildModule(PhotonTracer* rt, std::weak_ptr<Scene> scene);
        ~PhotonRebuildModule();

        // forward() will trigger a ray trace and return an image tensor
        torch::Tensor forward(PhotonTracer::RenderSettings& settings);

        float* getRenderedImage();
        void freeData();

        void uploadPathTracerFromTensor();

        GPUData m_data; // Todo rename
        GPUDataTensors m_tensorData;

    private:
        PhotonTracer* m_photonRebuild;
        void uploadFromScene(std::weak_ptr<Scene> scene);

    };

}


#endif //MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H
