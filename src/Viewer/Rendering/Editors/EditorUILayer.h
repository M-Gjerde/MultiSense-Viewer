//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_EDITORUILAYER
#define MULTISENSE_VIEWER_EDITORUILAYER

#include "Viewer/Rendering/ImGui/Layer.h"
#include "Viewer/Rendering/ImGui/IconsFontAwesome6.h"
#include "EditorDefinitions.h"

#include "Viewer/Application/Application.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class EditorUILayer : public Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        void togglePopup() {
            static bool toggles[] = {true, false, false, false, false};
            // Options for the combo box

            std::vector<EditorType> items = getSelectableEditorTypes();
            static int item_current_idx = 0; // Here we store our current item index
            m_editor->ui()->newEditorTypeRequested = false;

            m_editor->ui()->newEditorTypeRequested = false;
            if (ImGui::BeginPopup("EditorSelectionPopup")) {
                ImGui::SeparatorText("Editor Types");
                for (int i = 0; i < items.size(); i++) {
                    if (ImGui::Selectable(editorTypeToString(items[i]).c_str())) {
                        item_current_idx = i;
                        m_editor->ui()->selectedType = items[item_current_idx];
                        m_editor->ui()->newEditorTypeRequested = true;
                    }
                }
                ImGui::MenuItem("Console", nullptr, &m_editor->ui()->showDebugWindow);
                ImGui::EndPopup();
            }

        }

        /** Called once per frame **/
        void onUIRender() override {
            float borderSize = m_editor->ui()->layoutConstants.borderSize;

            {
                // Set window position and size
                ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
                ImVec2 window_size = ImVec2(m_editor->ui()->width, m_editor->ui()->height); // Size (width, height)


                // Set window flags to remove decorations
                ImGuiWindowFlags window_flags =
                        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                        ImGuiWindowFlags_NoFocusOnAppearing |
                        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;

                // Set next window position and size
                ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
                ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

                // Create the parent window
                ImGui::Begin("EditorBorderWindow", nullptr, window_flags);


                // End the parent window
                ImVec4 color;
                if (m_editor->ui()->active)
                    color = m_editor->ui()->backgroundColorActive;
                else if (m_editor->ui()->hovered)
                    color = m_editor->ui()->backgroundColorHovered;
                else
                    color = m_editor->ui()->backgroundColor;

                color.w = 1.0f;
                color = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
                ImU32 border_color = ImGui::ColorConvertFloat4ToU32(color); // White color

                ImDrawList *draw_list = ImGui::GetWindowDrawList();
                ImVec2 p1 = window_pos;
                ImVec2 p2 = ImVec2(window_pos.x + window_size.x, window_pos.y + borderSize);
                draw_list->AddRectFilled(p1, p2, border_color); // Top border
                ImVec2 p4 = ImVec2(window_pos.x + borderSize, window_pos.y + window_size.y);
                draw_list->AddRectFilled(p1, p4, border_color); // Left border

                draw_list->AddRectFilled(ImVec2(window_pos.x, window_size.y - borderSize),
                                         ImVec2(window_size.x, window_size.y), border_color); // Bottom border

                draw_list->AddRectFilled(ImVec2(window_size.x - borderSize, window_pos.y),
                                         ImVec2(window_size.x, window_size.y), border_color); // Right border


                ImGui::BeginChild("ButtonContainer", ImVec2(0, 0), false, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);

                // Make the button background transparent
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));           // Transparent background
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));    // Transparent when hovered
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 1, 0, 1));     // Red when active (clicked)


                //ImGui::PushFont(m_editor->guiResources().fontIcons);
                ImVec2 txtSize = ImGui::CalcTextSize("..");
                float yMax = ImGui::GetContentRegionAvail().y;
                ImGui::SetCursorPosY(yMax - txtSize.y);
                if (ImGui::Button("..", ImVec2(txtSize))) {
                }
                //ImGui::PopFont();
                ImGui::PopStyleColor(3);  // Pop the three color styles
                ImGui::EndChild();

                ImGui::End();
                ImGui::PopStyleVar(2);

            }

            {
                ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
                ImVec2 window_size = ImVec2(m_editor->ui()->layoutConstants.uiWidth,  m_editor->ui()->layoutConstants.uiHeight); // Size (width, height)

                // Set window flags to remove decorations
                ImGuiWindowFlags window_flags =
                        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground| ImGuiWindowFlags_NoBringToFrontOnFocus;

                // Set next window position and size
                ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
                ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);


                ImGui::Begin("EditorSelectorWindow", nullptr, window_flags);

                ImGui::PushFont(m_editor->guiResources().fontIcons);
                if (ImGui::Button(ICON_FA_WINDOW_RESTORE)) {
                    ImGui::OpenPopup("EditorSelectionPopup");
                }
                ImGui::PopFont();
                togglePopup();

                ImGui::End();
            }

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_EDITORUILAYER
