//
// Created by magnus on 1/2/25.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGEUI_H
#define MULTISENSE_VIEWER_EDITORIMAGEUI_H

#include "Viewer/Rendering/ImGui/Layer.h"

namespace VkRender {

    struct EditorImageUI : public EditorUI {

        bool uploadScene = false;
        bool render = false;
        bool saveImage = false;
        bool denoise = false;

        bool clearImageMemory = false;

        std::string kernel = "Path Tracer: Mesh";
        std::string kernelDevice = "GPU";
        bool switchKernelDevice = false;

        int photonCount = 1e6;
        int numBounces = 32;

        struct ShaderSelection {
            int someVariable = 0;
            float gammaCorrection = 2.2f;

        }shaderSelection;
        // Constructor that copies everything from base EditorUI
        EditorImageUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

}
#endif //MULTISENSE_VIEWER_EDITORIMAGEUI_H
