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

        std::string kernelDevice = "";
        int selectedDeviceIndex = 1;

        bool denoise = false;

        bool automatic = true;
        // Constructor that copies everything from base EditorUI
        EditorDifferentiableRendererLayerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

}
#endif //MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER_UI
