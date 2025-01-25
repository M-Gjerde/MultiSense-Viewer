//
// Created by magnus on 1/24/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PathTracerModule.h"

namespace VkRender {


    torch::Tensor
    PhotonRebuildModuleImpl::forward(EditorPathTracerLayerUI &uiLayer, std::shared_ptr<Scene> scene) {
        // 1) Call PhotonRebuild::update(...) or any function you want to do the actual path tracing
        //    In your case:  rt_->update(...);
        m_photonRebuild->update(uiLayer, scene);
        torch::Tensor someTensorParam;
        // Simply call the custom autograd function
        auto result = PathTracer::PhotonRebuildFunction::apply(
            someTensorParam,
            uiLayer,
            scene,
            m_photonRebuild
        );
        return result;
    }
}