//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER
#define MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER


#include "EditorDifferentiableRendererLayerUI.h"
#include "Viewer/Rendering/ImGui/Layer.h"

namespace VkRender {
    class EditorDifferentiableRendererLayer : public Layer {
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

            auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_editor->ui());

            imageUI->reloadRenderer = ImGui::Button("Reload");
            ImGui::SameLine();

            imageUI->step = ImGui::Button("Step");
            ImGui::SameLine();

            ImGui::Checkbox("Step##Toggle", &imageUI->toggleStep);


            ImGui::End();
        }

        /** Called once upon this object destruction **/
        void onDetach() override {
        }
    };
}

#endif //MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER
