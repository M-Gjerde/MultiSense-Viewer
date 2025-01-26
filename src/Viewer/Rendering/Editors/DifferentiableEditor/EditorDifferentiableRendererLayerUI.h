//
// Created by magnus on 1/2/25.
//

#ifndef MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER_UI
#define MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER_UI

#include "Viewer/Rendering/ImGui/Layer.h"

namespace VkRender {

    struct EditorDifferentiableRendererLayerUI : public EditorUI {
        bool reloadRenderer = false;
        bool step = false;
        bool toggleStep = false;
        bool backprop = false;
        bool uploadScene = false;

        bool checkStartAccumulation = true;

        std::string kernelDevice = "";
        int selectedDeviceIndex = 0;
        bool switchKernelDevice = false;

        // Old
        bool renderFromViewpoint = false;
        bool saveImage = false;
        bool denoise = false;

        // Render settings
        int photonCount = 1e4;
        int numBounces = 32;
        float gammaCorrection = 3.5f;

        // Constructor that copies everything from base EditorUI
        EditorDifferentiableRendererLayerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

}
#endif //MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER_UI
