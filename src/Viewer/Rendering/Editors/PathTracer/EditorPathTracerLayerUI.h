//
// Created by magnus on 1/2/25.
//

#ifndef MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
#define MULTISENSE_VIEWER_PATHTRACERLAYERUI_H

#include "Viewer/Rendering/ImGui/Layer.h"

namespace VkRender {

    struct EditorPathTracerLayerUI : public EditorUI {

        bool renderFromViewpoint = false;

        bool uploadScene = false;
        bool render = false;
        bool toggleRendering = false;
        bool saveImage = false;
        bool denoise = false;

        bool clearImageMemory = false;

        std::string kernel = " ";
        std::string kernelDevice = "CPU";
        bool switchKernelDevice = false;

        int photonCount = 1e4;
        int numBounces = 32;

        struct ShaderSelection {
            int someVariable = 0;
            float gammaCorrection = 2.2f;

        }shaderSelection;
        // Constructor that copies everything from base EditorUI
        EditorPathTracerLayerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

}
#endif //MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
