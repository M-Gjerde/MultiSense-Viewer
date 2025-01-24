//
// Created by magnus on 1/24/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PathTracerModule.h"

namespace VkRender {


    torch::Tensor
    VkRender::PhotonRebuildModuleImpl::forward(EditorPathTracerLayerUI &uiLayer, std::shared_ptr<Scene> scene) {
        // 1) Call PhotonRebuild::update(...) or any function you want to do the actual path tracing
        //    In your case:  rt_->update(...);
        m_photonRebuild->update(uiLayer, scene);

        // 2) Retrieve the float* from PhotonRebuild::getImage()
        float *image_ptr = m_photonRebuild->getImage();
        // This pointer must contain at least width_*height_ floats (or width_*height_*channels if RGBA).
        // We'll assume it's just a float buffer of size (width_*height_) or (width_*height_*4).

        // 3) Convert raw float* memory to a torch::Tensor
        //    Suppose your PhotonRebuild writes one float per pixel -> shape [height, width]
        //    or you might have 3 or 4 channels -> shape [height, width, 3(or 4)].

        const int channels = 1;  // or 3 or 4, depending on your PhotonRebuild
        // If your code actually is storing 1 float per pixel:
        // auto sizes = std::vector<int64_t>{ height_, width_ };
        // If 3 or 4 floats per pixel:
        auto sizes = std::vector<int64_t>{m_height, m_width, channels};

        // We use from_blob to wrap the existing memory.
        // from_blob does NOT take ownership, so we typically .clone() afterwards to get a safe tensor.
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto tensor = torch::from_blob(image_ptr, sizes, options).clone();

        // 4) Return the cloned tensor
        return tensor;
    }
}