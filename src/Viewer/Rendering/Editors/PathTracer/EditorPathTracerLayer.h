//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H
#define MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H


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
            ImVec2 window_pos = ImVec2( m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
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

            ImGui::Checkbox("Active camera", &imageUI->renderFromViewpoint);
            ImGui::SameLine();

            imageUI->uploadScene = ImGui::Button("UploadScene");
            ImGui::SameLine();

            imageUI->render = ImGui::Button("Render");
            ImGui::SameLine();
            ImGui::Checkbox("Toggle Render", &imageUI->toggleRendering);

            ImGui::SameLine();

            ImGui::Checkbox("Denoise", &imageUI->denoise);

            ImGui::SameLine();

            // Dropdown for selecting render kernel
            const char* kernels[] = { "Hit-Test", "Path Tracer: Mesh" , "Path Tracer: 2DGS" };

            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("##Render Kernel", &imageUI->selectedKernelIndex, kernels, IM_ARRAYSIZE(kernels))) {
            }

            imageUI->kernel = kernels[imageUI->selectedKernelIndex];

            ImGui::SameLine();            // Dropdown for selecting render kernel
            const char* selections[] = { "CPU", "GPU" }; // TODO This should come from selectSyclDevices
            static int selectedDevieType = 0; // Default selection
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("##Select Device Type", &selectedDevieType, selections, IM_ARRAYSIZE(selections))) {
                imageUI->switchKernelDevice = true;
            }
            imageUI->kernelDevice = selections[selectedDevieType];

            ImGui::SameLine();

            imageUI->clearImageMemory = ImGui::Button("Clear image memory");
            if (imageUI->clearImageMemory){
                imageUI->render = true;
            }

            const int sliderMin = 1000;
            const int sliderMax = 10000000;

            ImGui::SetNextItemWidth(150);
            if (ImGui::SliderInt("PhotonCount", &imageUI->photonCount, sliderMin, sliderMax, "%d", ImGuiSliderFlags_Logarithmic)) {
                // Normalize to the nearest 10,000 and ensure it's at least 1000
                imageUI->photonCount = std::max((imageUI->photonCount + 5000) / 10000 * 10000, sliderMin);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            if(ImGui::SliderInt("Light Bounces", &imageUI->numBounces, 1, 100)){
                imageUI->clearImageMemory = true;
            };
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat("Gamma", &imageUI->shaderSelection.gammaCorrection, 0, 8);

            ImGui::SameLine();
            imageUI->saveImage = ImGui::Button("Save");


            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}

#endif //MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H
