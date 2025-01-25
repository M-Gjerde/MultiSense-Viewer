//
// Created by magnus on 8/14/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DLAYER_H
#define MULTISENSE_VIEWER_EDITOR3DLAYER_H


#include "Viewer/Rendering/ImGui/Layer.h"
#include "Viewer/Rendering/ImGui/IconsFontAwesome6.h"
#include "Viewer/Rendering/Editors/EditorDefinitions.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {
    enum class OutputTextureImageType { Color, Depth };

    enum class DepthColorOption : int32_t {
        None,
        Invert,
        Normalize,
        JetColormap,
        ViridisColormap
    };

    struct Editor3DViewportUI : EditorUI {
        bool renderFromViewpoint = false;
        bool saveNextFrame = false;
        // Image type selection
        OutputTextureImageType selectedImageType = OutputTextureImageType::Color;
        // Depth color option selection (only relevant if Depth is selected)
        DepthColorOption depthColorOption = DepthColorOption::None;
        bool reloadViewportShader = false;
        // Constructor that copies everything from base EditorUI
        explicit Editor3DViewportUI(const EditorUI &baseUI) : EditorUI(baseUI) {
        }
    };

    class Editor3DLayer : public Layer {
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
            ImVec2 window_pos = ImVec2(m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width - window_pos.x,
                                        m_editor->ui()->height - window_pos.y); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("Editor3DLayer", nullptr, window_flags);

            auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_editor->ui());
            auto editor = reinterpret_cast<Editor3DViewport *>(m_editor);

            ImGui::Checkbox("Active camera", &imageUI->renderFromViewpoint);
            ImGui::SameLine();
            imageUI->saveNextFrame = ImGui::Button("Save");
            ImGui::SameLine();
            imageUI->reloadViewportShader = ImGui::Button("Reload Shader");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::BeginCombo("Image Type",
                                  imageUI->selectedImageType == OutputTextureImageType::Color ? "Color" : "Depth")) {
                if (ImGui::Selectable("Color", imageUI->selectedImageType == OutputTextureImageType::Color)) {
                    if (imageUI->selectedImageType != OutputTextureImageType::Color) {
                        imageUI->selectedImageType = OutputTextureImageType::Color;
                        editor->onRenderSettingsChanged();
                    }
                }
                if (ImGui::Selectable("Depth", imageUI->selectedImageType == OutputTextureImageType::Depth)) {
                    if (imageUI->selectedImageType != OutputTextureImageType::Depth) {
                        imageUI->selectedImageType = OutputTextureImageType::Depth;
                        editor->onRenderSettingsChanged();
                        imageUI->depthColorOption = DepthColorOption::Invert;
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SameLine();

            // Show Depth options if Depth is selected
            if (imageUI->selectedImageType == OutputTextureImageType::Depth) {
                ImGui::Text("Depth Color Options");
                ImGui::SameLine();

                ImGui::SetNextItemWidth(100.0f);
                if (ImGui::BeginCombo("Color Option",
                                      imageUI->depthColorOption == DepthColorOption::None
                                          ? "None" : imageUI->depthColorOption == DepthColorOption::Invert
                                          ? "Invert"
                                          : imageUI->depthColorOption == DepthColorOption::Normalize
                                                ? "Normalize"
                                                : imageUI->depthColorOption == DepthColorOption::JetColormap
                                                      ? "Colormap (Jet)"
                                                      : "Colormap (Viridis)")) {
                    if (ImGui::Selectable("Invert", imageUI->depthColorOption == DepthColorOption::Invert)) {
                        if (imageUI->depthColorOption != DepthColorOption::Invert) {
                            imageUI->depthColorOption = DepthColorOption::Invert;
                            editor->onRenderSettingsChanged();
                        }
                    }
                    if (ImGui::Selectable("Normalize", imageUI->depthColorOption == DepthColorOption::Normalize)) {
                        if (imageUI->depthColorOption != DepthColorOption::Normalize) {
                            imageUI->depthColorOption = DepthColorOption::Normalize;
                            editor->onRenderSettingsChanged();
                        }
                    }
                    if (ImGui::Selectable("Colormap (Jet)",
                                          imageUI->depthColorOption == DepthColorOption::JetColormap)) {
                        if (imageUI->depthColorOption != DepthColorOption::JetColormap) {
                            imageUI->depthColorOption = DepthColorOption::JetColormap;
                            editor->onRenderSettingsChanged();
                        }
                    }
                    if (ImGui::Selectable("Colormap (Viridis)",
                                          imageUI->depthColorOption == DepthColorOption::ViridisColormap)) {
                        if (imageUI->depthColorOption != DepthColorOption::ViridisColormap) {
                            imageUI->depthColorOption = DepthColorOption::ViridisColormap;
                            editor->onRenderSettingsChanged();
                        }
                    }
                    ImGui::EndCombo();
                }
            } else {
                imageUI->depthColorOption = DepthColorOption::None;
            }

            ImGui::End();
        }

        /** Called once upon this object destruction **/
        void onDetach()
        override {
        }
    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DLAYER_H
