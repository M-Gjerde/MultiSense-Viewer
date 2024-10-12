//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGERLAYER
#define MULTISENSE_VIEWER_EDITORIMAGERLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"

namespace VkRender {

    struct EditorImageUI : public EditorUI {
        bool renderMultiSense = false;

        bool renderFromSceneCamera = false;
        bool update = false;

        int previewID = 0;
        std::string selectedCameraName = "";

        bool playVideoFromFolder = false;

        // Constructor that copies everything from base EditorUI
        EditorImageUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

    class EditorImageLayer : public Layer {

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
            ImGui::Begin("EditorImageLayer", nullptr, window_flags);

            auto imageUI = std::dynamic_pointer_cast<EditorImageUI>(m_editor->ui());


            ImGui::Checkbox("Scene camera", &imageUI->renderFromSceneCamera); ImGui::SameLine();
            if (imageUI->renderFromSceneCamera) {
                auto view = m_scene->getRegistry().view<CameraComponent>();
                std::vector<std::string> cameraEntityNames;
                for (auto entity : view) {
                    if (m_scene->getRegistry().all_of<TagComponent>(entity)) {
                        auto& tag = m_scene->getRegistry().get<TagComponent>(entity);
                        cameraEntityNames.push_back(tag.Tag);  // Assuming TagComponent has a 'name' string member
                    }
                }
                const char* combo_preview_value = cameraEntityNames[imageUI->previewID].c_str(); // Preview value for the combo box
                // Step 3: Create the combo box with ImGui
                ImGui::SetNextItemWidth(150.0f);
                if (ImGui::BeginCombo(("Camera ##" + m_editor->getUUID().operator std::string()).c_str(), combo_preview_value)) {
                    for (int n = 0; n < cameraEntityNames.size(); n++) {
                        const bool is_selected = (imageUI->previewID == n);
                        if (ImGui::Selectable(cameraEntityNames[n].c_str(), is_selected)) {
                            imageUI->previewID = n;
                            imageUI->selectedCameraName = cameraEntityNames[n];
                            imageUI->update = true;
                            auto* editor = reinterpret_cast<EditorImage *>(m_editor);
                        }

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                    ImGui::SameLine();
                }
            }
            ImGui::SetNextItemWidth(150.0f);
            // If we have a device connected:
            if (m_context->multiSense()->anyMultiSenseDeviceOnline()) {

                // Available sources from camera
                auto& profile = m_context->multiSense()->getSelectedMultiSenseProfile();
                std::vector<std::string>& availableSources = profile.deviceData().sources;
                int& currentItemIndex = profile.deviceData().streamWindow[m_editor->getUUID()].selectedSourceIndex;

                // Ensure there is at least one option
                if (availableSources.empty()) {
                    availableSources.emplace_back("Error: No sources available");
                    currentItemIndex = 0;
                }
                const char* combo_preview_value = availableSources[currentItemIndex].c_str();  // Pass in the preview value visible before opening the combo (it could be anything)

                if (ImGui::BeginCombo(("combo 1##" + m_editor->getUUID().operator std::string()).c_str(), combo_preview_value, ImGuiComboFlags_None))
                {
                    for (int n = 0; n < availableSources.size(); n++)
                    {
                        const bool is_selected = (currentItemIndex == n);
                        if (ImGui::Selectable(availableSources[n].c_str(), is_selected)) {
                            currentItemIndex = n;
                            profile.deviceData().streamWindow[m_editor->getUUID()].enabledSource = availableSources[n];
                            profile.deviceData().streamWindow[m_editor->getUUID()].sourceUpdate = true;
                        }
                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Play video from folder", &imageUI->playVideoFromFolder);
            ImGui::SameLine();
            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}
#endif //MULTISENSE_VIEWER_EDITORIMAGERLAYER
