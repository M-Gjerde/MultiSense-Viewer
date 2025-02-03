//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H
#define MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H


#include <Viewer/Rendering/RenderResources/PathTracer/Definitions.h>

#include "Viewer/Rendering/ImGui/Layer.h"
#include "Viewer/Rendering/ImGui/IconsFontAwesome6.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"

namespace VkRender {
    class EditorPathTracerLayer : public Layer {
    public:
        /** Called once upon this object creation**/
        void onAttach() override {
        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {
        }


        /** Called once per frame **/
        void onUIRender() override {
            // Set window position and size
            // Set window position and size
            ImVec2 window_pos = ImVec2(m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width - window_pos.x,
                                        m_editor->ui()->height - window_pos.y); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            // Create the parent window
            ImGui::Begin("EditorPathTracerLayer", nullptr, window_flags);

            auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_editor->ui());

            imageUI->resetPathTracer = ImGui::Button("Reset");
            ImGui::SameLine();


            ImGui::Checkbox("Scene Camera", &imageUI->useSceneCamera);
            ImGui::SameLine();


            imageUI->render = ImGui::Button("Render");
            ImGui::SameLine();
            ImGui::Checkbox("Toggle Render", &imageUI->toggleRendering);

            ImGui::SameLine();

            ImGui::Checkbox("Denoise", &imageUI->denoise);

            ImGui::SameLine();


            // Prepare dropdown items
            const char* kernels[PathTracer::KERNEL_TYPE_COUNT];

            for (int i = 0; i < PathTracer::KERNEL_TYPE_COUNT; ++i) {
                kernels[i] = PathTracer::KernelTypeToString(static_cast<PathTracer::KernelType>(i));
            }
            // Render ImGui combo box
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("##Render Kernel", &imageUI->selectedKernelIndex, kernels,
                             PathTracer::KERNEL_TYPE_COUNT)) {
                // Update the kernel based on selection
            }
            imageUI->kernel = static_cast<PathTracer::KernelType>(imageUI->selectedKernelIndex);


            ImGui::SameLine(); // Dropdown for selecting render kernel
            const char* selections[] = {"CPU", "GPU"}; // TODO This should come from selectSyclDevices
            imageUI->switchKernelDevice = false;
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("##Select Device Type", &imageUI->selectedDeviceIndex, selections,
                             IM_ARRAYSIZE(selections))) {
                imageUI->switchKernelDevice = true;
            }
            imageUI->kernelDevice = selections[imageUI->selectedDeviceIndex];

            ImGui::SameLine();

            imageUI->clearImageMemory = ImGui::Button("Clear image memory");

            const int sliderMin = 1000;
            const int sliderMax = 10000000;

            ImGui::SetNextItemWidth(150);
            if (ImGui::SliderInt("PhotonCount", &imageUI->photonCount, sliderMin, sliderMax, "%d",
                                 ImGuiSliderFlags_Logarithmic)) {
                // Normalize to the nearest 10,000 and ensure it's at least 1000
                imageUI->photonCount = std::max((imageUI->photonCount + 5000) / 10000 * 10000, sliderMin);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            if (ImGui::SliderInt("Light Bounces", &imageUI->numBounces, 1, 100)) {
                imageUI->clearImageMemory = true;
            };
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat("Gamma", &imageUI->shaderSelection.gammaCorrection, 0, 8);

            ImGui::SameLine();
            imageUI->saveImage = ImGui::Button("Save");

            // new row
            EditorPathTracer* editor = static_cast<EditorPathTracer*>(m_editor);
            ImGui::Text("Frame Number: %u", editor->getRenderInformation().frameID);


            ImGui::End();
        }

        /** Called once upon this object destruction **/
        void onDetach() override {
        }
    };
}

#endif //MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H
