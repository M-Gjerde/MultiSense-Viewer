//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include "Layer.h"
#include "imgui_user.h"

class InteractionMenu : public Layer {
public:

// Create global object for convenience in other functions

    void onFinishedRender() override {

    }


    void OnUIRender(GuiObjectHandles *handles) override {
        //if (handles->devices->empty()) return;

        // Check if stream was interrupted by a disconnect event and reset pages events across all devices
        for (auto &d: *handles->devices) {
            for (auto &s: d.streams)
                if (s.second.playbackStatus == AR_PREVIEW_RESET) {
                    s.second.playbackStatus = AR_PREVIEW_NONE;
                    std::fill_n(page, PAGE_TOTAL_PAGES, false);
                    drawActionPage = true;
                }
        }

        for (int i = 0; i < PAGE_TOTAL_PAGES; ++i) {
            if (page[i]) {
                if (drawActionPage) {
                    drawActionPage = false;
                }
                switch (i) {
                    case PAGE_PREVIEW_DEVICES:
                        buildPreview(handles);
                        break;
                    case PAGE_DEVICE_INFORMATION:
                        buildDeviceInformation(handles);
                        break;
                    case PAGE_CONFIGURE_DEVICE:
                        buildConfigurationPreview(handles);
                        break;
                }
            }
        }

        if (drawActionPage) {
            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
            ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f));
            ImGui::Begin("InteractionMenu", &pOpen, window_flags);

            int imageButtonHeight = 100;
            const char *labels[3] = {"Preview Device", "Device Information", "Configure Device"};
            for (int i = 0; i < PAGE_TOTAL_PAGES; i++) {
                float offset = 150;
                ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + (i * offset),
                                           (handles->info->height / 2) - (imageButtonHeight / 2)));

                ImGui::PushID(i);
                ImVec2 size = ImVec2(100.0f, 100.0f);                     // Size of the image we want to make visible
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = ImVec4(0.054, 0.137, 0.231, 1.0f);         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                if (ImGui::ImageButton(handles->info->imageButtonTextureDescriptor[i], size, uv0, uv1, 0, bg_col,
                                       tint_col))
                    page[i] = true;
                ImGui::PopID();

                ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + (i * offset),
                                           (handles->info->height / 2) + (imageButtonHeight / 2) + 8));

                ImGui::Text("%s", labels[i]);

                ImGui::SameLine();
            }
            ImGui::NewLine();
            //ImGui::ShowDemoWindow();
            ImGui::End();
            ImGui::PopStyleColor(); // bg color
        }
    }


    void addDropDown(GuiObjectHandles *handles, std::string id, StreamIndex streamIndex, StreamingModes *stream) {


        ImVec2 position = ImGui::GetCursorScreenPos();
        position.x += 0;//handles->info->sidebarWidth;
        ImVec2 pos = position;
        pos.x += 250.0f;
        pos.y += 30.0f;


        if (openDropDown[streamIndex]) {
            pos = position;
            if (animationLength[streamIndex] < 220)
                animationLength[streamIndex] += 1;
            pos.y += animationLength[streamIndex];
            pos.x += 250;
        } else {
            animationLength[streamIndex] = 0;
        }

        ImGui::GetWindowDrawList()->AddRectFilled(position, pos, ImColor(0.17, 0.157, 0.271, 1.0f), 10.0f, 0);
        ImGui::CustomSelectable(id.c_str(), &openDropDown[streamIndex], 0, ImVec2(250.0f, 30.0f));

        if (openDropDown[streamIndex]) {

            ImGui::Dummy(ImVec2(0.0f, 10.0f));

            ImGui::Text("Data source:");

            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::BeginCombo(std::string("##Source" + std::to_string(streamIndex)).c_str(),
                                  stream->sources[stream->selectedSourceIndex].c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (int n = 0; n < stream->sources.size(); n++) {

                    const bool is_selected = (stream->selectedSourceIndex == n);
                    if (ImGui::Selectable(stream->sources[n].c_str(), is_selected)) {
                        stream->selectedSourceIndex = n;
                        stream->selectedStreamingSource = stream->sources[stream->selectedSourceIndex];

                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::Text("Resolution:");

            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::BeginCombo(std::string("##Resolution" + std::to_string(streamIndex)).c_str(),
                                  stream->modes[stream->selectedModeIndex].c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (int n = 0; n < stream->modes.size(); n++) {
                    const bool is_selected = (stream->selectedModeIndex == n);
                    if (ImGui::Selectable(stream->modes[n].c_str(), is_selected)) {
                        stream->selectedModeIndex = n;
                        stream->selectedStreamingMode = stream->modes[stream->selectedModeIndex];

                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Dummy(ImVec2(0.0f, 30.0f));

            std::string btnLabel = "Start##" + std::to_string(streamIndex);
            if (ImGui::Button(btnLabel.c_str())) {
                stream->playbackStatus = AR_PREVIEW_PLAYING;
            }
            ImGui::SameLine();
            btnLabel = "Pause##" + std::to_string(streamIndex);
            if (ImGui::Button(btnLabel.c_str())) {
                stream->playbackStatus = AR_PREVIEW_PAUSED;
            }
            btnLabel = "Stop##" + std::to_string(streamIndex);
            if (ImGui::Button(btnLabel.c_str())) {
                stream->playbackStatus = AR_PREVIEW_NONE;
            }

            ImGui::Dummy(ImVec2(0.0f, 40.0f));


            //openDropDownMenu(handles, position);
        }
    }

private:
    bool page[PAGE_TOTAL_PAGES] = {false, false, false};
    bool drawActionPage = true;
    bool openDropDown[5] = {false};
    float animationLength[5] = {false};

    void openDropDownMenu(GuiObjectHandles *handles, ImVec2 position) {

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.17, 0.157, 0.271, 1.0f));
        position.y += 30.0f;
        position.x += 200.0f;
        ImGui::SetNextWindowPos(position, 0);
        ImGui::BeginChild("child", ImVec2(200.0f, 150.0f));

        ImGui::Text("Select data source");

        const char *items[] = {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ",
                               "KKKK", "LLLLLLL", "MMMM", "OOOOOOO"};
        static int item_current_idx = 0; // Here we store our selection data as an index.
        const char *combo_preview_value = items[item_current_idx];  // Pass in the preview value visible before opening the combo (it could be anything)

        if (ImGui::BeginCombo("Custom", combo_preview_value, 0)) {
            for (int n = 0; n < IM_ARRAYSIZE(items); n++) {

                const bool is_selected = (item_current_idx == n);
                if (ImGui::Selectable(items[n], is_selected)) {
                    item_current_idx = n;

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Text("Select resolution");

        const char *items2[] = {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ",
                                "KKKK", "LLLLLLL", "MMMM", "OOOOOOO"};
        static int item_current_idx2 = 0; // Here we store our selection data as an index.
        const char *combo_preview_value2 = items2[item_current_idx2];  // Pass in the preview value visible before opening the combo (it could be anything)

        if (ImGui::BeginCombo("Custom2", combo_preview_value2, 0)) {
            for (int n = 0; n < IM_ARRAYSIZE(items2); n++) {

                const bool is_selected = (item_current_idx2 == n);
                if (ImGui::Selectable(items2[n], is_selected)) {
                    item_current_idx2 = n;

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }


        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    void buildDeviceInformation(GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_DEVICE_INFORMATION] = false;
            drawActionPage = true;
        }

        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }

    void buildPreview(GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 0.0f));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_PREVIEW_DEVICES] = false;
            drawActionPage = true;
        }

        ImVec2 pos = ImGui::GetCursorPos();
        pos.y += 5;
        pos.x += 50;
        ImGui::SetCursorPos(pos);

        for (auto &d: *handles->devices) {
            if (d.cameraName == "Virtual Camera") {
                addStreamPlaybackControls(AR_PREVIEW_VIRTUAL, "Virtual preview", &d);
            }
        }
        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }


    void buildConfigurationPreview(GuiObjectHandles *handles) {
        for (auto& dev: *handles->devices) {
            if (dev.state != ArActiveState)
                continue;

            // Control page
            ImGui::BeginGroup();
            createControlArea(handles, dev);
            ImGui::EndGroup();

            // Viewing page
            ImGui::BeginGroup();
            createViewingArea(handles, dev);
            ImGui::EndGroup();

        }
        //ImGui::PopStyleColor(); // bg color
        //ImGui::End();
    }

    void createViewingArea(GuiObjectHandles *handles, Element &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(
                ImVec2(handles->info->sidebarWidth + handles->info->offset5px + handles->info->controlAreaWidth +
                       handles->info->offset5px, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.034, 0.107, 0.201, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(handles->info->viewingAreaWidth, handles->info->viewingAreaHeight));
        ImGui::Begin("ViewingArea", &pOpen, window_flags);


        ImGui::Dummy(ImVec2((handles->info->viewingAreaWidth / 2 ) - 30.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PushFont(handles->info->font18);
        ImGui::Text("Viewing");
        ImGui::PopFont();

        ImGui::Dummy(ImVec2((handles->info->viewingAreaWidth / 2 )- 80.0f, 0.0f));
        ImGui::SameLine();
        if (ImGui::Button("2D", ImVec2(75.0f, 20.0f)))
            dev.selectedPreviewTab = TAB_2D_PREVIEW;
        ImGui::SameLine();
        if(ImGui::Button("3D", ImVec2(75.0f, 20.0f)))
            dev.selectedPreviewTab = TAB_3D_POINT_CLOUD;

        ImGui::End();
        ImGui::PopStyleColor();
    }

    void createControlArea(GuiObjectHandles *handles, Element &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth + handles->info->offset5px, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.034, 0.107, 0.201, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(handles->info->controlAreaWidth, handles->info->controlAreaHeight));
        ImGui::Begin("ControlArea", &pOpen, window_flags);

        for (auto &d: *handles->devices) {
            // Create dropdown
            if (d.state == ArActiveState) {
                ImGuiTabBarFlags tab_bar_flags = 0;//ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                    if (ImGui::BeginTabItem("Control")) {

                        if (ImGui::Button("Back")) {
                            page[PAGE_CONFIGURE_DEVICE] = false;
                            drawActionPage = true;
                        }

                        // Create Control Area
                        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth + handles->info->offset5px +
                                                       (handles->info->offset5px * 2), handles->info->tabAreaHeight),
                                                ImGuiCond_Always);
                        ImGui::BeginChild("Dropdown Area", ImVec2(handles->info->controlAreaWidth,
                                                                  handles->info->height - handles->info->tabAreaHeight),
                                          false, ImGuiCond_Always);
                        StreamingModes *stream = nullptr;
                        for (auto &d: *handles->devices) {
                            if (d.state == ArActiveState) {
                                addDropDown(handles, "1. Left Camera", AR_PREVIEW_LEFT, &d.streams[AR_PREVIEW_LEFT]);
                                addDropDown(handles, "2. Right Camera", AR_PREVIEW_RIGHT, &d.streams[AR_PREVIEW_RIGHT]);
                                addDropDown(handles, "3. Auxiliary Camera", AR_PREVIEW_AUXILIARY,
                                            &d.streams[AR_PREVIEW_AUXILIARY]);
                                addDropDown(handles, "4. Disparity", AR_PREVIEW_DISPARITY,
                                            &d.streams[AR_PREVIEW_DISPARITY]);

                                addDropDown(handles, "5. Point Cloud", AR_PREVIEW_POINT_CLOUD,
                                            &d.streams[AR_PREVIEW_POINT_CLOUD]);


                            }
                        }


                        ImGui::EndChild();


                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Configuration")) {

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Future Tab 1")) {

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Future Tab 2")) {

                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
        }
        ImGui::PopStyleColor();
        ImGui::End();
    }


    void addStreamPlaybackControls(CameraStreamInfoFlag streamIndex, std::string label, Element *d) {
        ImGui::BeginGroup();
        ImGui::Text("%s", label.c_str());
        auto &stream = d->streams[streamIndex];
        ImGui::SetNextItemWidth(200);
        const char *combo_preview_value = stream.sources[stream.selectedSourceIndex].c_str();  // Pass in the preview value visible before opening the combo (it could be anything)
        std::string srcLabel = "##Source" + std::to_string(streamIndex);
        if (ImGui::BeginCombo(srcLabel.c_str(), combo_preview_value,
                              ImGuiComboFlags_HeightSmall)) {
            for (int n = 0; n < stream.sources.size(); n++) {

                const bool is_selected = (stream.selectedSourceIndex == n);
                if (ImGui::Selectable(stream.sources[n].c_str(), is_selected)) {
                    stream.selectedSourceIndex = n;
                    stream.selectedStreamingSource = stream.sources[stream.selectedSourceIndex];

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SetNextItemWidth(200);

        std::string resLabel = "##Resolution" + std::to_string(streamIndex);
        if (ImGui::BeginCombo(resLabel.c_str(), stream.modes[stream.selectedModeIndex].c_str(),
                              ImGuiComboFlags_HeightSmall)) {
            for (int n = 0; n < stream.modes.size(); n++) {
                const bool is_selected = (stream.selectedModeIndex == n);
                if (ImGui::Selectable(stream.modes[n].c_str(), is_selected)) {
                    stream.selectedModeIndex = n;
                    stream.selectedStreamingMode = stream.modes[stream.selectedModeIndex];

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        std::string btnLabel = "Play/Pause##" + std::to_string(streamIndex);
        if (ImGui::Button(btnLabel.c_str())) {
            if (stream.playbackStatus != AR_PREVIEW_PLAYING)
                stream.playbackStatus = AR_PREVIEW_PLAYING;
            else
                stream.playbackStatus = AR_PREVIEW_STOPPED;
        }

        ImGui::EndGroup();
    }


};

#endif //MULTISENSE_INTERACTIONMENU_H
