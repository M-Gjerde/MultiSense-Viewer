//
// Created by magnus on 1/2/25.
//

#ifndef MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
#define MULTISENSE_VIEWER_PATHTRACERLAYERUI_H

#include "Viewer/Rendering/ImGui/Layer.h"

#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"
#include "Viewer/Rendering/Editors/EditorIncludes.h"

namespace VkRender {

    struct EditorPathTracerLayerUI : EditorUI {

        bool useSceneCamera = false;

        bool uploadScene = false;
        bool render = false;
        bool toggleRendering = false;
        bool saveImage = false;
        bool bypassSave = false;
        bool denoise = false;

        bool clearImageMemory = false;

        PathTracer::KernelType kernel = PathTracer::KERNEL_PATH_TRACER_2DGS;
        std::string kernelDevice = "";
        int selectedKernelIndex = PathTracer::KERNEL_PATH_TRACER_2DGS;
        int selectedDeviceIndex = 0;
        bool switchKernelDevice = false;

        int photonCount = 1e4;
        int numBounces = 0;

        bool resetPathTracer = false;

        struct ShaderSelection {
            int someVariable = 0;
            float gammaCorrection = 2.2f;

        }shaderSelection;
        // Constructor that copies everything from base EditorUI
        EditorPathTracerLayerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

}
#endif //MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
