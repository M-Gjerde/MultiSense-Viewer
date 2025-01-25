//
// Created by magnus on 1/24/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildModule.h"

namespace VkRender::PathTracer {

    PhotonRebuildModule::PhotonRebuildModule(PhotonTracer* rt)
            : m_photonRebuild(rt) {
        // Optionally register parameters or buffers if needed

        myParam = register_parameter(
    "myParam",
    torch::randn({1}, torch::requires_grad(true))
);
    }

    torch::Tensor
    PhotonRebuildModule::forward(PhotonTracer::Settings &settings, std::shared_ptr<Scene> scene) {
        // 1) Call PhotonTracer::update(...) or any function you want to do the actual path tracing
        //    In your case:  rt_->update(...);

        // Simply call the custom autograd function
        auto result = PathTracer::PhotonRebuildFunction::apply(
            myParam,
            settings,
            scene,
            m_photonRebuild
        );
        return result;
    }

    float* PhotonRebuildModule::getRenderedImage()
    {
        return m_photonRebuild->getImage();
    }
}
