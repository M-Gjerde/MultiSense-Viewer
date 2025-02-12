/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/NewVersionAvailable.h
 *
 * Copyright 2023
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2023-05-19, mgjerde@carnegierobotics.com, Created file.
 **/


#ifndef MULTISENSE_VIEWER_NEWVERSIONAVAILABLE_H
#define MULTISENSE_VIEWER_NEWVERSIONAVAILABLE_H

#include "Viewer/ImGui/Layer.h"


void openURL(const std::string &url) {
#ifdef _WIN32
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
#elif __linux__
    std::string command = "xdg-open " + std::string(url);
    int result = std::system(command.c_str());
    if(result != 0) {
        Log::Logger::getInstance()->warning("Failed top open URL");
    }

#endif
}

class NewVersionAvailable : public VkRender::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *uiHandle) override {
        // We dont want to risk blocking usage permissions modal with NewVersionAvailable popup

        if (uiHandle->newVersionAvailable && uiHandle->askUserForNewVersion) {
            ImGui::OpenPopup("New Version Available!", ImGuiPopupFlags_NoOpenOverExistingPopup);
        } else {
            return;
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 5.0f));
        ImVec2 anonymousWindowSize(500.0f, 180.0f);
        ImGui::SetNextWindowPos(ImVec2((uiHandle->info->width / 2) - (anonymousWindowSize.x / 2),
                                       (uiHandle->info->height / 2) - (anonymousWindowSize.y / 2) - 50.0f));
        if (ImGui::BeginPopupModal("New Version Available!", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            std::string url = "https://github.com/carnegierobotics/multisense_viewer/releases";
            static bool isLinkHovered = false;
            ImVec4 blueLinkColor = isLinkHovered ? ImVec4(0.17f, 0.579f, 0.893f, 1.0f) : ImVec4(0.0f, 0.439f, 0.753f,
                                                                                                1.0f);

            ImGui::Text("Dear User, a shiny new version of this viewer is ready for you! \n");
            ImGui::Text("Upgrade now and enjoy new features and improvements.\n");
            ImGui::Text("New version can be found by clicking the button or here: \n");

            ImGui::PushStyleColor(ImGuiCol_Text, blueLinkColor);
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0)); // Transparent button background
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                                  ImVec4(0, 0, 0, 0)); // Transparent button background when hovered
            ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                                  ImVec4(0, 0, 0, 0)); // Transparent button background when active
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::CalcTextSize("GitHub Releases").x);
            if (ImGui::Selectable("GitHub Releases", false, ImGuiSelectableFlags_NoAutoClosePopups)) {
                openURL(url);
                uiHandle->usageMonitor->userClickAction("GitHub Releases", "Selectable", ImGui::GetCurrentWindow()->Name);

            }

            isLinkHovered = ImGui::IsItemHovered();
            ImGui::PopStyleColor(4);
            ImGui::Spacing();
            ImGui::Spacing();

            if (ImGui::Button("Awesome, Let's go!")) {
                openURL(url);
                uiHandle->askUserForNewVersion = false;
                uiHandle->usageMonitor->userClickAction("Awesome, Let's go!", "Button", ImGui::GetCurrentWindow()->Name);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SetItemDefaultFocus();

            ImGui::SameLine();

            if (ImGui::Button("Remind me later")) {
                uiHandle->usageMonitor->userClickAction("Remind me later", "Button", ImGui::GetCurrentWindow()->Name);

                uiHandle->askUserForNewVersion = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(2);
    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }
};

#endif //MULTISENSE_VIEWER_NEWVERSIONAVAILABLE_H
