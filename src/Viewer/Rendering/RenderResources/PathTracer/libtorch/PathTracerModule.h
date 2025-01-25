//
// Created by magnus on 1/24/25.
//

#ifndef MULTISENSE_VIEWER_PATHTRACERMODULE_H
#define MULTISENSE_VIEWER_PATHTRACERMODULE_H

#include <torch/torch.h>

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildFunction.h"
// Wrap your raytracer in a Torch module
namespace VkRender {
    class PhotonRebuildModuleImpl : public torch::nn::Module {
    public:
        PhotonRebuildModuleImpl(std::shared_ptr<PathTracer::PhotonRebuild> rt, int64_t width, int64_t height)
                : m_photonRebuild(std::move(rt)), m_width(width), m_height(height) {
            // Optionally register parameters or buffers if needed
        }

        // forward() will trigger a ray trace and return an image tensor
        torch::Tensor forward(EditorPathTracerLayerUI &uiLayer, std::shared_ptr<Scene> scene);

    private:
        std::shared_ptr<PathTracer::PhotonRebuild> m_photonRebuild;
        int64_t m_width, m_height;
    };

}

#endif //MULTISENSE_VIEWER_PATHTRACERMODULE_H
