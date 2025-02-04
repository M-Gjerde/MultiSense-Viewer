/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/SideBar.h
 *
 * Copyright 2022
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
 *   2022-4-19, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_SIDEBAR_H
#define MULTISENSE_SIDEBAR_H

#include <algorithm>
#include <queue>

#define IMGUI_DEFINE_MATH_OPERATORS
DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
#include <imgui.h>
#include <imgui_internal.h>
DISABLE_WARNING_POP

#include <sys/types.h>

#include "Viewer/Tools/Utils.h"
#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/AdapterUtils.h"
#include "Viewer/Tools/ReadSharedMemory.h"

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <shellapi.h>

#define AutoConnectReader ReaderWindows
#define elevated() Utils::hasAdminRights()
#else

#include <unistd.h>
#include <sys/types.h>
#include <cstdlib>

#define AutoConnectReader ReaderLinux
#define elevated() getuid()
#endif


class SideBarLayer : public VkRender::Layer
{
public:
    typedef enum LOG_COLOR
    {
        LOG_COLOR_GRAY = 0,
        LOG_COLOR_GREEN = 1,
        LOG_COLOR_RED = 2
    } LOG_COLOR;

    VkRender::EntryConnectDevice m_Entry;
    std::vector<VkRender::EntryConnectDevice> entryConnectDeviceList;
    uint32_t gifFrameIndex = 0;
    uint32_t gifFrameIndex2 = 0;
    std::chrono::steady_clock::time_point gifFrameTimer;
    std::chrono::steady_clock::time_point gifFrameTimer2;
    std::chrono::steady_clock::time_point searchingTextAnimTimer;
    std::vector<AdapterUtils::Adapter> manualConnectAdapters;
    AdapterUtils adapterUtils;
    std::string dots;
    bool btnConnect = false;
    bool enableConnectButton = true;

    enum
    {
        MANUAL_CONNECT = 1,
        AUTO_CONNECT = 2
    };

    // Create global object for convenience in other functions
    std::vector<std::string> interfaceNameList;
    std::vector<std::string> interfaceIDList; // Windows required
    std::vector<uint32_t> indexList;

    int connectMethodSelector = MANUAL_CONNECT;
    ImGuiTextBuffer Buf;
    ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
    ImVector<LOG_COLOR> colors;
    ImVec4 lastLogTextColor = VkRender::Colors::TextColorGray;
    size_t ethernetComboIndex = 0;
    int resultsComboIndex = -1;

    bool startedAutoConnect = false;
    std::unique_ptr<AutoConnectReader> reader;
#ifdef __linux__
    FILE *autoConnectProcess = nullptr;
#else
    SHELLEXECUTEINFOA shellInfo;
#endif
    std::string btnLabel = "Start";

    void onAttach() override
    {
        gifFrameTimer = std::chrono::steady_clock::now();
        gifFrameTimer2 = std::chrono::steady_clock::now();
    }

    void onFinishedRender() override
    {
    }

    void onDetach() override
    {
        adapterUtils.stopAdapterScan();
#ifdef __linux__
        if (autoConnectProcess != nullptr) {
            if (reader)
                reader->sendStopSignal();
            pclose(autoConnectProcess);
#else
        if (shellInfo.hProcess != nullptr)
        {
            if (reader)
                reader->sendStopSignal();
            if (TerminateProcess(shellInfo.hProcess, 1) != 0)
            {
                Log::Logger::getInstance()->info("Terminated AutoConnect program");
            }
            else
                Log::Logger::getInstance()->info("Failed to terminate AutoConnect program");
#endif
        }
        auto startTime = std::chrono::steady_clock::now();
        // Make sure adapter utils scanning thread is shut down correctly
        while (true)
        {
            Log::Logger::getInstance()->traceWithFrequency("shutdown adapterutils: ", 1000,
                                                           "Waiting for adapterUtils to shutdown");
            if (adapterUtils.shutdownReady())
                break;
        }

        auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
            std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Adapter utils took {}s to shut down", timeSpan.count());
    }

    static void openURL(const std::string& url)
    {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif __linux__
        std::string command = "xdg-open " + std::string(url);
        int result = std::system(command.c_str());
        if(result != 0) {
            Log::Logger::getInstance()->warning("Failed top open URL");
        }
#endif
    }

    static void askUsageLoggingPermissionPopUp(VkRender::GuiObjectHandles* handle)
    {
        if (VkRender::RendererConfig::getInstance().getUserSetting().askForUsageLoggingPermissions)
        {
            ImGui::OpenPopup("Anonymous Usage Statistics");

            auto user = VkRender::RendererConfig::getInstance().getUserSetting();
            user.askForUsageLoggingPermissions = false;
            VkRender::RendererConfig::getInstance().setUserSetting(user);
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 5.0f));
        ImVec2 anonymousWindowSize(500.0f, 180.0f);
        ImGui::SetNextWindowPos(ImVec2((handle->info->width / 2) - (anonymousWindowSize.x / 2),
                                       (handle->info->height / 2) - (anonymousWindowSize.y / 2) - 50.0f));
        if (ImGui::BeginPopupModal("Anonymous Usage Statistics", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            std::string url =
                "https://github.com/carnegierobotics/multisense_viewer/blob/master/Assets/Generated/PrivacyPolicy.md";
            static bool isLinkHovered = false;
            ImVec4 blueLinkColor = isLinkHovered
                                       ? ImVec4(0.17f, 0.579f, 0.893f, 1.0f)
                                       : ImVec4(0.0f, 0.439f, 0.753f,
                                                1.0f);

            ImGui::Text("We would like to collect anonymous usage statistics to help improve our product.");
            ImGui::Text("Data collected will only be used for product improvement purposes");
            ImGui::Text("More information can be found at: ");
            ImGui::PushStyleColor(ImGuiCol_Text, blueLinkColor);
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0)); // Transparent button background
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                                  ImVec4(0, 0, 0, 0)); // Transparent button background when hovered
            ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                                  ImVec4(0, 0, 0, 0)); // Transparent button background when active
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::CalcTextSize("Privacy policy").x);
            if (ImGui::Selectable("Privacy policy", false, ImGuiSelectableFlags_NoAutoClosePopups))
            {
                openURL(url);
                handle->usageMonitor->userClickAction("Privacy policy", "Selectable",
                                                      ImGui::GetCurrentWindow()->Name);
            }
            isLinkHovered = ImGui::IsItemHovered();
            ImGui::PopStyleColor(4);

            ImGui::Spacing();
            ImGui::Text("Do you grant us permission to log and collect anonymous usage statistics?");
            ImGui::Text("This option can always be changed in the settings tab");
            auto user = VkRender::RendererConfig::getInstance().getUserSetting();
            static int radio_value = user.userConsentToSendLogs;
            bool update = ImGui::RadioButton("Yes", &radio_value, 1);
            ImGui::SameLine();
            update |= ImGui::RadioButton("No", &radio_value, 0);
            if (update)
            {
                user.userConsentToSendLogs = radio_value;
                VkRender::RendererConfig::getInstance().setUserSetting(user);
                handle->usageMonitor->setSetting("user_consent_to_collect_statistics", radio_value ? "true" : "false");
                handle->usageMonitor->userClickAction("Yes|No", "RadioButton",
                                                      ImGui::GetCurrentWindow()->Name);
            }

            ImVec2 btnSize(120.0f, 0.0f);
            ImGui::SetCursorPosX((ImGui::GetWindowWidth() / 2) - (btnSize.x / 2));
            if (ImGui::Button("OK", btnSize))
            {
                handle->usageMonitor->setSetting("ask_user_consent_to_collect_statistics", "false");
                user.userConsentToSendLogs = radio_value;
                VkRender::RendererConfig::getInstance().setUserSetting(user);
                handle->usageMonitor->setSetting("user_consent_to_collect_statistics", radio_value ? "true" : "false");

                handle->usageMonitor->userClickAction("OK", "button",
                                                      ImGui::GetCurrentWindow()->Name);

                ImGui::CloseCurrentPopup();
            }

            ImGui::SetItemDefaultFocus();
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(2);
    }

    void onUIRender(VkRender::GuiObjectHandles* handles) override
    {
        if (handles->renderer3D)
        {
            return;
        }

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth, handles->info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLGray424Main);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 10.0f));
        ImGui::Begin("SideBar", &pOpen, window_flags);
        addPopup(handles);
        askUsageLoggingPermissionPopUp(handles);
        ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        if (ImGui::Button("Settings", ImVec2(handles->info->sidebarWidth, 17.0f)))
        {
            handles->showDebugWindow = !handles->showDebugWindow;
            handles->usageMonitor->userClickAction("Settings", "button", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::PopStyleVar();


        if (!handles->devices.empty())
            sidebarElements(handles);
        addDeviceButton(handles);

        // Add version number
        ImGui::SetCursorPos(ImVec2((handles->info->sidebarWidth / 2) - (handles->info->addDeviceWidth / 2),
                                   handles->info->height - (handles->info->addDeviceBottomPadding) + 35.0f));
        ImGui::PushFont(handles->info->font8);
        ImGui::Text("%s", (std::string("Ver: ") + VkRender::RendererConfig::getInstance().getAppVersion()).c_str());
        ImGui::PopFont();
        ImGui::End();
        ImGui::PopStyleColor(); // bg color
        ImGui::PopStyleVar(2);
    }

private:
    void addLogLine(LOG_COLOR color, const char* fmt, ...) IM_FMTARGS(3) {
        int old_size = Buf.size();
        va_list args;
        va_start(args, fmt);
        Buf.appendfv(fmt, args);
        va_end(args);
        for (int new_size = Buf.size(); old_size < new_size; old_size++)
            if (Buf[old_size] == '\n')
            {
                LineOffsets.push_back(old_size + 1);
            }
        colors.push_back(color);
    }

    /**@brief Function to manage Auto connect with GUI updates
     * Requirements:
     * Should run once popup modal is opened
     * Periodically check the status of connected ethernet devices
     * */
    void autoDetectCamera()
    {
        // if reader is not opened we don't start autoconnect program
        if (!reader)
            return;

        if (startedAutoConnect)
        {
            // Try to open reader. TODO timeout or number of tries for opening
            if (!reader->isOpen)
                reader->open();
            else
            {
                if (reader->read())
                {
                    std::string str = reader->getLogLine();
                    if (!str.empty())
                        addLogLine(LOG_COLOR_GRAY, "%s", str.c_str());
                    if (reader->stopRequested)
                    {
                        reader->sendStopSignal();

                        m_Entry.IP.clear();
                        resultsComboIndex = -1;
                        entryConnectDeviceList.clear();

#ifdef __linux__
                        if (autoConnectProcess != nullptr && pclose(autoConnectProcess) == 0) {
                            startedAutoConnect = false;
                            reader.reset();
                            Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                            autoConnectProcess = nullptr;
                            btnLabel = "Start";
                            return;
                        }
#else
                        if (shellInfo.hProcess != nullptr)
                        {
                            if (TerminateProcess(shellInfo.hProcess, 1) != 0)
                            {
                                Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                            }
                            shellInfo.hProcess = nullptr;
                            reader.reset();
                            btnLabel = "Start";
                            startedAutoConnect = false;
                            return;
                        }
#endif
                    }
                    // Same IP on same adapter is treated as the same camera
                    std::vector<VkRender::EntryConnectDevice> entries = reader->getResult();
                    for (const auto& entry : entries)
                    {
                        if (!entry.cameraName.empty())
                        {
                            bool cameraExists = false;
                            for (const auto& e : entryConnectDeviceList)
                            {
                                if (e.IP == entry.IP && e.interfaceIndex == entry.interfaceIndex)
                                    cameraExists = true;
                            }
                            if (!cameraExists)
                            {
                                VkRender::EntryConnectDevice e{
                                    entry.IP, entry.interfaceName, entry.cameraName,
                                    entry.interfaceIndex,
                                    entry.description
                                };

                                entryConnectDeviceList.push_back(e);
                            }
                        }
                    }
                }
            }
        }
        else
        {
#ifdef __linux__
            std::string fileName = "./AutoConnectLauncher.sh";
            autoConnectProcess = popen((fileName).c_str(), "r");
            if (autoConnectProcess == nullptr) {
                Log::Logger::getInstance()->info("Failed to start new process, error: %s", strerror(errno));
            } else {
                startedAutoConnect = true;
            }
#else
            std::string fileName = ".\\AutoConnect.exe";
            shellInfo.lpVerb = "runas";
            shellInfo.cbSize = sizeof(SHELLEXECUTEINFO);
            shellInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
            shellInfo.hwnd = nullptr;
            LPCSTR winFileName = fileName.c_str();
            shellInfo.lpFile = winFileName;
            shellInfo.lpParameters = "-i on -c on";
            shellInfo.lpDirectory = ".\\";
            shellInfo.nShow = SW_HIDE;
            shellInfo.hInstApp = nullptr;
            bool instance = ShellExecuteExA(&shellInfo);

            if (!instance)
            {
                Log::Logger::getInstance()->info("Failed to start new process, error: %zu", GetLastError());
                reader.reset();
                btnLabel = "Start";
                // Reset attempts
            }
            else
            {
                startedAutoConnect = true;
            }
#endif
        }
    }

    static void createDefaultElement(VkRender::GuiObjectHandles* handles, const VkRender::EntryConnectDevice& entry,
                                     bool fromWindowsAutoConnect = false)
    {
        VkRender::Device el{};

        el.name = entry.profileName;
        el.IP = entry.IP;
        el.state = fromWindowsAutoConnect ? VkRender::CRL_STATE_JUST_ADDED_WINDOWS : VkRender::CRL_STATE_JUST_ADDED;
        Log::Logger::getInstance()->info("Set dev {}'s state to VkRender::CRL_STATE_JUST_ADDED. On Windows? {} ",
                                         el.name,
                                         fromWindowsAutoConnect);
        el.interfaceName = entry.interfaceName;
        el.interfaceDescription = entry.description;
        el.clicked = true;
        el.interfaceIndex = entry.interfaceIndex;
        el.isRemoteHead = entry.isRemoteHead;
        handles->devices.push_back(el);

        Log::Logger::getInstance()->info("Connect clicked for Default Device");
        Log::Logger::getInstance()->info("Using: Ip: {}, and profile: {} for {}", entry.IP, entry.profileName,
                                         entry.description);
    }


    void sidebarElements(VkRender::GuiObjectHandles* handles)
    {
        auto& devices = handles->devices;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            auto& e = devices.at(i);
            std::string buttonIdentifier;
            ImVec4 btnColor{};
            // Set colors based on state
            switch (e.state)
            {
            case VkRender::CRL_STATE_CONNECTED:
                break;
            case VkRender::CRL_STATE_CONNECTING:
                buttonIdentifier = "Connecting";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
                btnColor = VkRender::Colors::CRLBlueIsh;
                break;
            case VkRender::CRL_STATE_ACTIVE:
                buttonIdentifier = "Active";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray421);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.42f, 0.31f, 1.0f));
                btnColor = ImVec4(0.26f, 0.42f, 0.31f, 1.0f);
                break;
            case VkRender::CRL_STATE_INACTIVE:
                buttonIdentifier = "Inactive";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLRed);
                btnColor = VkRender::Colors::CRLRed;
                break;
            case VkRender::CRL_STATE_LOST_CONNECTION:
                buttonIdentifier = "Lost connection...";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
                btnColor = VkRender::Colors::CRLBlueIsh;
                break;
            case VkRender::CRL_STATE_DISCONNECTED:
                buttonIdentifier = "Disconnected";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLRed);
                btnColor = VkRender::Colors::CRLRed;
                break;
            case VkRender::CRL_STATE_UNAVAILABLE:
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLDarkGray425);
                btnColor = VkRender::Colors::CRLDarkGray425;
                buttonIdentifier = "Unavailable";
                break;
            case VkRender::CRL_STATE_JUST_ADDED:
            case VkRender::CRL_STATE_JUST_ADDED_WINDOWS:
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                btnColor = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
                buttonIdentifier = "Added...";
                break;
            case VkRender::CRL_STATE_DISCONNECT_AND_FORGET:
                buttonIdentifier = "Disconnecting...cmake";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                break;
            case VkRender::CRL_STATE_REMOVE_FROM_LIST:
                buttonIdentifier = "Removing...";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                break;
            case VkRender::CRL_STATE_RESET:
                buttonIdentifier = "Resetting";
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                break;
            default:
                break;
            }

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            std::string winId = e.name + "Child";
            ImGui::BeginChild(winId.c_str(), ImVec2(handles->info->sidebarWidth, handles->info->elementHeight),
                              false, ImGuiWindowFlags_NoDecoration);


            // Stop execution here
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);

            // Delete a profile
            {
                if (ImGui::SmallButton("X"))
                {
                    // delete and disconnect devices
                    if (handles->devices.at(i).state == VkRender::CRL_STATE_CONNECTING)
                        handles->devices.at(i).state = VkRender::CRL_STATE_INTERRUPT_CONNECTION;
                    else
                        handles->devices.at(i).state = VkRender::CRL_STATE_DISCONNECT_AND_FORGET;

                    Log::Logger::getInstance()->info("Set dev {}'s state to VkRender::CRL_STATE_DISCONNECT_AND_FORGET ",
                                                     handles->devices.at(i).name);
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor(3);
                    ImGui::EndChild();
                    continue;
                }
            }
            ImGui::PopStyleColor();

            ImGui::SetCursorPos(ImVec2(0.0f, ImGui::GetCursorPosY()));

            ImGui::SameLine();
            ImGui::Dummy(ImVec2(0.0f, 20.0f));


            ImVec2 window_pos = ImGui::GetWindowPos();
            ImVec2 window_size = ImGui::GetWindowSize();
            ImVec2 window_center = ImVec2(window_pos.x + window_size.x * 0.5f, window_pos.y + window_size.y * 0.5f);
            ImVec2 cursorPos = ImGui::GetCursorPos();
            ImVec2 lineSize;
            // Profile Name
            {
                ImGui::PushFont(handles->info->font24);
                lineSize = ImGui::CalcTextSize(e.name.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(cursorPos);
                ImGui::Text("%s", e.name.c_str());
                ImGui::PopFont();
            }
            // Camera Name
            {
                ImGui::PushFont(handles->info->font13);
                lineSize = ImGui::CalcTextSize(e.cameraName.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
                ImGui::Text("%s", e.cameraName.c_str());
                ImGui::PopFont();
            }
            // Camera IP Address
            {
                ImGui::PushFont(handles->info->font13);
                lineSize = ImGui::CalcTextSize(e.IP.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));

                ImGui::Text("%s", e.IP.c_str());
                ImGui::PopFont();
            }

            // Status Button
            {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::PushFont(handles->info->font18);
                //ImGuiStyle style = ImGui::GetStyle();
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
                cursorPos.x = window_center.x - (ImGui::GetFontSize() * 10 / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
            }

            //buttonIdentifier += "##" + e.IP;
            //e.clicked = ImGui::Button(buttonIdentifier.c_str(), ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2));

            ImVec2 uv0 = ImVec2(0.0f, 0.0f); // UV coordinates for lower-left
            ImVec2 uv1 = ImVec2(1.0f, 1.0f);
            ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 0.3f); // No tint

            // Check if other device is already attempting to connect
            bool busy = false;
            for (const auto& dev : handles->devices)
            {
                if (dev.state == VkRender::CRL_STATE_CONNECTING)
                    busy = true;
            }

            // Connect button
            if (buttonIdentifier == "Connecting")
            {
                e.clicked = ImGui::ButtonWithGif(buttonIdentifier.c_str(), ImVec2(ImGui::GetFontSize() * 10, 35.0f),
                                                 handles->info->gif.image[gifFrameIndex2], ImVec2(35.0f, 35.0f),
                                                 uv0,
                                                 uv1,
                                                 tint_col, btnColor) && !busy;
            }
            else
            {
                e.clicked = ImGui::Button(buttonIdentifier.c_str(),
                                          ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2)) && !busy;
            }
            auto time = std::chrono::steady_clock::now();
            auto time_span =
                std::chrono::duration_cast<std::chrono::duration<float>>(time - gifFrameTimer2);

            if (time_span.count() > static_cast<float>(*handles->info->gif.delay) / 1000.0f)
            {
                gifFrameTimer2 = std::chrono::steady_clock::now();
                gifFrameIndex2++;
            }

            if (gifFrameIndex2 >= handles->info->gif.totalFrames)
                gifFrameIndex2 = 0;

            ImGui::PopFont();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(2);

            ImGui::EndChild();
        }
    }

    static void addDeviceButton(VkRender::GuiObjectHandles* handles)
    {
        ImGui::SetCursorPos(ImVec2((handles->info->sidebarWidth / 2) - (handles->info->addDeviceWidth / 2),
                                   handles->info->height - handles->info->addDeviceBottomPadding));

        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
        if (ImGui::Button("ADD DEVICE", ImVec2(handles->info->addDeviceWidth, handles->info->addDeviceHeight)) ||
            handles->openAddDevicePopup)
        {
            ImGui::OpenPopup("add_device_modal");
            handles->usageMonitor->userClickAction("ADD_DEVICE", "button", ImGui::GetCurrentWindow()->Name);
            handles->openAddDevicePopup = false;
        }
        ImGui::PopStyleColor();
    }

    void addPopup(VkRender::GuiObjectHandles* handles)
    {
        ImGui::SetNextWindowSize(ImVec2(handles->info->popupWidth, handles->info->popupHeight), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLCoolGray);


        if (ImGui::BeginPopupModal("add_device_modal", nullptr,
                                   ImGuiWindowFlags_NoDecoration))
        {
            /** HEADER FIELD */
            ImVec2 popupDrawPos = ImGui::GetCursorScreenPos();
            ImVec2 headerPosMax = popupDrawPos;
            headerPosMax.x += handles->info->popupWidth;
            headerPosMax.y += 50.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                      ImColor(VkRender::Colors::CRLRed), 9.0f, 0);
            popupDrawPos.y += 40.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                      ImColor(VkRender::Colors::CRLRed), 0.0f, 0);

            ImGui::PushFont(handles->info->font24);
            std::string title = "Connect to MultiSense";
            ImVec2 size = ImGui::CalcTextSize(title.c_str());
            float anchorPoint =
                (handles->info->popupWidth - size.x) / 2; // Make a m_Title in center of popup window


            ImGui::Dummy(ImVec2(0.0f, size.y));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 10.0f);

            ImGui::SetCursorPosX(anchorPoint);
            ImGui::Text("%s", title.c_str());
            ImGui::PopFont();

            ImGui::Dummy(ImVec2(0.0f, 25.0f));

            /** PROFILE NAME FIELD */

            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::Text("Profile Name:");
            ImGui::PopStyleColor();
            ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::Colors::CRLDarkGray425);
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
            ImGui::CustomInputTextWithHint("##InputProfileName", "MultiSense Profile", &m_Entry.profileName,
                                           ImGuiInputTextFlags_AutoSelectAll);

            /** SELECT METHOD FOR CONNECTION FIELD */
            /*
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::Text("2. Select method for connection:");
            ImGui::PopStyleColor();
            ImGui::Dummy(ImVec2(0.0f, 5.0f));

            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(10.0f, 0.0f));

            ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
            ImVec2 uv1 = ImVec2(1.0f, 1.0f);
            ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            //ImGui::BeginChild("IconChild", ImVec2(handles->info->popupWidth, 40.0f), false, ImGuiWindowFlags_NoDecoration);
            ImVec2 imageButtonSize(245.0f, 55.0f);
            ImGui::PushFont(handles->info->font15);
            if (ImGui::ImageButtonText("Automatic", &connectMethodSelector, AUTO_CONNECT, imageButtonSize,
                                       handles->info->imageButtonTextureDescriptor[3], ImVec2(33.0f, 31.0f), uv0, uv1,
                                       tint_col)) {
                Log::Logger::getInstance()->info(
                        "User clicked AUTO_CONNECT. Tab is {}, 0 = none, 1 = AutoConnect, 2 = ManualConnect",
                        connectMethodSelector);
                handles->usageMonitor->userClickAction("Automatic", "ImageButtonText", ImGui::GetCurrentWindow()->Name);

            }
            ImGui::SameLine(0, 30.0f);
            if (ImGui::ImageButtonText("Manual", &connectMethodSelector, MANUAL_CONNECT, imageButtonSize,
                                       handles->info->imageButtonTextureDescriptor[4], ImVec2(40.0f, 40.0f), uv0, uv1,
                                       tint_col)) {
                Log::Logger::getInstance()->info(
                        "User clicked MANUAL_CONNECT. Tab is {}, 0 = none, 1 = AutoConnect, 2 = ManualConnect",
                        connectMethodSelector);
                handles->usageMonitor->userClickAction("Manual", "ImageButtonText", ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopFont();

            ImGui::PopStyleVar();
            //ImGui::EndChild();

            ImGui::PopStyleVar(2); // RadioButton
            autoDetectCamera();

            */
            ImGui::PopStyleColor(); // ImGuiCol_FrameBg

            /** AUTOCONNECT FIELD BEGINS HERE*/
            if (connectMethodSelector == AUTO_CONNECT)
            {
                adapterUtils.stopAdapterScan(); // Stop scan if we select manual as autoconnect will run its own scan
                m_Entry.cameraName = "AutoConnect";
                ImGui::Dummy(ImVec2(0.0f, 12.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();

                if (ImGui::Button(btnLabel.c_str(), ImVec2(80.0f, 20.0f)))
                {
                    Log::Logger::getInstance()->info("User clicked {}", btnLabel);
                    if (btnLabel == "Start")
                    {
                        handles->usageMonitor->userClickAction("Start", "button", ImGui::GetCurrentWindow()->Name);
                        reader = std::make_unique<AutoConnectReader>();
                        btnLabel = "Reset";
                    }
                    else
                    {
                        handles->usageMonitor->userClickAction("Reset", "button", ImGui::GetCurrentWindow()->Name);
                        reader->sendStopSignal();
                        entryConnectDeviceList.clear();
                        m_Entry.IP.clear();
                        m_Entry.cameraName.clear();
                        resultsComboIndex = -1;
#ifdef __linux__
                        if (autoConnectProcess != nullptr) {
                            int err = pclose(autoConnectProcess);
                            if (err != 0) {
                                Log::Logger::getInstance()->info(
                                        "Error in closing AutoConnectProcess err: {} errno: {}", err, strerror(errno));

                            } else {
                                Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                            }
                            startedAutoConnect = false;
                            reader.reset();
                            autoConnectProcess = nullptr;
                            btnLabel = "Start";
                        }
#else
                        if (shellInfo.hProcess != nullptr)
                        {
                            if (TerminateProcess(shellInfo.hProcess, 1) != 0)
                            {
                                Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                            }
                            shellInfo.hProcess = nullptr;
                            reader.reset();
                            btnLabel = "Start";
                            startedAutoConnect = false;
                        }
#endif
                    }
                }


                /** STATUS SPINNER */
                ImGui::SameLine();
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 12.0f);
                // Create child window regardless of gif spinner state in order to keep cursor m_Position constant
                ImGui::BeginChild("Gif viewer", ImVec2(40.0f, 40.0f), false, ImGuiWindowFlags_NoDecoration);
                if (startedAutoConnect)
                    addSpinnerGif(handles);
                ImGui::EndChild();

                ImGui::SameLine(0, 360.0f);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLBlueIsh);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker(" If no packet at adapter is received try the following: \n "
                    " 1. Reconnect ethernet cables \n "
                    " 2. Power cycle the camera \n "
                    " 3. Wait 20-30 seconds. If no packet is received contact support  \n\n");

                ImGui::PopStyleColor(2);

                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLDarkGray425);
                const char* id = "Log Window";
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));

                ImGui::SameLine();
                ImGui::BeginChild(id, ImVec2(handles->info->popupWidth - 40.0f, 85.0f), false, 0);
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0));
                const char* buf = Buf.begin();
                const char* buf_end = Buf.end();
                ImGuiListClipper clipper;
                clipper.Begin(LineOffsets.Size);
                while (clipper.Step())
                {
                    for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
                    {
                        const char* line_start = buf + LineOffsets[line_no];
                        const char* line_end = (line_no + 1 < LineOffsets.Size)
                                                   ? (buf + LineOffsets[line_no + 1] -
                                                       1)
                                                   : buf_end;
                        // Weird index magic. colors is an ImGui vector.
                        if (line_no > 0 && line_no < colors.size() - 1)
                        {
                            LOG_COLOR col = colors[line_no + 1];
                            switch (col)
                            {
                            case LOG_COLOR_GRAY:
                                lastLogTextColor = VkRender::Colors::TextColorGray;
                                break;
                            case LOG_COLOR_GREEN:
                                lastLogTextColor = VkRender::Colors::TextGreenColor;
                                break;
                            case LOG_COLOR_RED:
                                lastLogTextColor = VkRender::Colors::TextRedColor;
                                break;
                            default:
                                lastLogTextColor = VkRender::Colors::TextColorGray;
                                break;
                            }
                        }
                        ImGui::PushStyleColor(ImGuiCol_Text, lastLogTextColor);
                        ImGui::Dummy(ImVec2(5.0f, 0.0f));
                        ImGui::SameLine();
                        ImGui::TextUnformatted(line_start, line_end);
                        ImGui::PopStyleColor();
                    }
                }
                clipper.End();
                ImGui::PopStyleColor();


                ImGui::PopStyleVar();
                if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                    ImGui::SetScrollHereY(1.0f);
                ImGui::EndChild();
                ImGui::Dummy(ImVec2(0.0f, 30.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();

                auto time = std::chrono::steady_clock::now();
                auto time_span =
                    std::chrono::duration_cast<std::chrono::duration<float>>(time - searchingTextAnimTimer);

                if (time_span.count() > 0.35f && startedAutoConnect)
                {
                    searchingTextAnimTimer = std::chrono::steady_clock::now();
                    dots.append(".");

                    if (dots.size() > 4)
                        dots.clear();
                }
                else if (!startedAutoConnect)
                    dots = ".";

                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);

                if (entryConnectDeviceList.empty() && startedAutoConnect)
                {
                    ImGui::Text("%s", ("Searching" + dots).c_str());
                }
                else
                {
                    ImGui::Text("Select device:");
                }

                ImGui::PopStyleColor();
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                static bool selected = false;

                if (!selected)
                {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4());
                    m_Entry.interfaceName.clear();
                }
                else
                {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyle().Colors[ImGuiCol_Header]);
                }
                // Header
                // HeaderHovered
                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLDarkGray425);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 5.0f));

                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::BeginChild("##ResultsChild", ImVec2(handles->info->popupWidth - (20.0f * 2.0f), 50.0f), true,
                                  0);
                for (size_t n = 0; n < entryConnectDeviceList.size(); n++)
                {
                    if (entryConnectDeviceList[n].cameraName.empty()) continue;
                    if (ImGui::Selectable((entryConnectDeviceList[n].cameraName + "##" + std::to_string(n)).c_str(),
                                          resultsComboIndex == static_cast<int>(n),
                                          ImGuiSelectableFlags_NoAutoClosePopups,
                                          ImVec2(handles->info->popupWidth - (20.0f * 2), 15.0f)))
                    {
                        handles->usageMonitor->userClickAction(entryConnectDeviceList[n].cameraName, "Selectable",
                                                               ImGui::GetCurrentWindow()->Name);

                        resultsComboIndex = static_cast<int>(n);
                        entryConnectDeviceList[n].profileName = entryConnectDeviceList[n].cameraName;
                        // Keep profile m_Name if user inputted this before auto-connect is finished
                        m_Entry = entryConnectDeviceList[n];
                        selected = true;
                    }
                }
                ImGui::EndChild();
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar();

                /*
                ImGui::Dummy(ImVec2(20.0f, 10.0));
                ImGui::Dummy(ImVec2(20.0f, 0.0));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                ImGui::Checkbox(" Remote Head", &m_Entry.isRemoteHead);
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLBlueIsh);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\n  Check this if you are connecting to a remote head device  \n ");
                ImGui::PopStyleColor(3);
                */
            }
            /** MANUAL_CONNECT FIELD BEGINS HERE*/
            else if (connectMethodSelector == MANUAL_CONNECT)
            {
                // AdapterSearch Threaded operation
                // Threaded adapter search for manual connect
#ifdef WIN32
                adapterUtils.startAdapterScan(handles->pool.get());
#endif

                {
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                    ImGui::Text("Camera IP:");
                    ImGui::PopStyleColor();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                }

                ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::Colors::CRLDarkGray425);
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                ImGui::CustomInputTextWithHint("##inputIP", "Default: 10.66.171.21", &m_Entry.IP,
                                               ImGuiInputTextFlags_CharsScientific |
                                               ImGuiInputTextFlags_AutoSelectAll |
                                               ImGuiInputTextFlags_CharsNoBlank);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));

                {
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                    ImGui::Text("Select network adapter:");
                    ImGui::PopStyleColor();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::SameLine(0.0f, 10.0f);
                }
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                // Call once a second
                manualConnectAdapters = adapterUtils.getAdaptersList();
                interfaceNameList.clear();
                indexList.clear();
                // Following three ifdef macros are a bit janky but required since Windows uses a HEX-ID AND Name to describe a network adapter whereas linux only uses a Name
#ifdef WIN32
                interfaceIDList.clear();
#endif

                // immediate mode vector item ordering
                // -- requirements --
                // Always have a base item in it.
                // if we push back other items then remove base item
                // No identical items
                for (const auto& a : manualConnectAdapters)
                {
#ifdef WIN32
                    if (a.supports && !Utils::isInVector(interfaceNameList, a.description))
                    {
                        interfaceNameList.push_back(a.description);
                        interfaceIDList.push_back(a.ifName);
#else
                    if (a.supports && !Utils::isInVector(interfaceNameList, a.ifName)) {
                        interfaceNameList.push_back(a.ifName);
#endif
                        indexList.push_back(a.ifIndex);
                        if (Utils::isInVector(interfaceNameList, "No adapters found"))
                        {
                            Utils::delFromVector(&interfaceNameList, std::string("No adapters found"));
                        }
                    }
                }
                if (!Utils::isInVector(interfaceNameList, "No adapters found") &&
                    interfaceNameList.empty())
                {
                    interfaceNameList.emplace_back("No adapters found");
                    indexList.push_back(0);
                }
                if (ethernetComboIndex >= interfaceNameList.size())
                    ethernetComboIndex = interfaceNameList.size() - 1;
#ifdef WIN32
                if (!interfaceIDList.empty())
                    m_Entry.interfaceName = interfaceIDList[ethernetComboIndex];
                std::string previewValue = interfaceNameList[ethernetComboIndex];
#else
                m_Entry.interfaceName = interfaceNameList[ethernetComboIndex];
                std::string previewValue = m_Entry.interfaceName;
#endif
                m_Entry.interfaceIndex = indexList[ethernetComboIndex];
                //if (!adapters.empty())
                //    m_Entry.interfaceName = adapters[ethernetComboIndex].networkAdapter;
                static ImGuiComboFlags flags = 0;
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLDarkGray425);
                if (ImGui::BeginCombo("##SelectAdapter", previewValue.c_str(), flags))
                {
                    for (size_t n = 0; n < interfaceNameList.size(); n++)
                    {
                        const bool is_selected = (ethernetComboIndex == n);
                        if (ImGui::Selectable(interfaceNameList[n].c_str(), is_selected))
                        {
                            ethernetComboIndex = static_cast<uint32_t>(n);
                            handles->usageMonitor->userClickAction("SelectAdapter", "combo",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }
                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopStyleColor(2); // ImGuiCol_FrameBg
                m_Entry.cameraName = "Manual";
                /*
                ImGui::Dummy(ImVec2(40.0f, 20.0));
                ImGui::Dummy(ImVec2(20.0f, 0.0));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                //ImGui::Checkbox("  Configure System Network", &handles->configureNetwork);
                //ImGui::SameLine(0, 20.0f);

                ImGui::Checkbox("  Remote Head", &m_Entry.isRemoteHead);
                ImGui::SameLine(0, 5.0f);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLBlueIsh);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\n  Check this if you are connecting to a remote head device  \n ");
                ImGui::PopStyleColor(3);
                 */

#ifdef WIN32
                std::string url = "https://docs.carnegierobotics.com/network/host_configuration/windows.html";
#else
                std::string url = "https://docs.carnegierobotics.com/network/host_configuration/linux.html";
#endif

                ImGui::Dummy(ImVec2(0.0f, 25.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::Text("Having trouble connecting to a camera?");
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::Text("Refer to our guide by clicking below for setting up the network configuration.");
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                static bool isLinkHovered = false;
                ImVec4 blueLinkColor = isLinkHovered
                                           ? ImVec4(0.17f, 0.579f, 0.893f, 1.0f)
                                           : ImVec4(0.0f, 0.439f, 0.753f,
                                                    1.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, blueLinkColor);
                std::string selectableTxt = "Network Configuration Guide";
                if (ImGui::Selectable(selectableTxt.c_str(), false, ImGuiSelectableFlags_NoAutoClosePopups,
                                      ImGui::CalcTextSize(selectableTxt.c_str())))
                {
                    openURL(url);
                    handles->usageMonitor->userClickAction("Network Configuration", "Selectable",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                ImGui::PopStyleColor(2);
            }
            else
            {
                adapterUtils.stopAdapterScan(); // Stop it if it was started and we deselect manual connect
            }

            if (handles->configureNetwork)
            {
                if (elevated() && connectMethodSelector == MANUAL_CONNECT)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLRed);
                    ImGui::Dummy(ImVec2(40.0f, 10.0));
                    ImGui::Dummy(ImVec2(20.0f, 0.0));
                    ImGui::SameLine();
#ifdef WIN32
                    ImGui::Text("Launch app as admin to \nauto configure network adapter");
#else
                    ImGui::Text("Launch app as root to \nauto configure network adapter");
#endif
                    enableConnectButton = false;
                    ImGui::PopStyleColor();
                }
                else
                    enableConnectButton = true;
            }
            else
                enableConnectButton = true;


            if (connectMethodSelector == AUTO_CONNECT && !reader)
            {
                enableConnectButton = false;
            }

            /** CANCEL/CONNECT FIELD BEGINS HERE*/
            ImGui::Dummy(ImVec2(0.0f, 40.0f));
            ImGui::SetCursorPos(ImVec2(0.0f, handles->info->popupHeight - 50.0f));
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::PushFont(handles->info->font15);
            bool btnCancel = ImGui::Button("Close", ImVec2(190.0f, 30.0f));
            ImGui::SameLine(0, 130.0f);
            if (!m_Entry.ready(handles->devices, m_Entry) || !enableConnectButton)
            {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::TextColorGray);
            }
            btnConnect = ImGui::Button("Connect", ImVec2(190.0f, 30.0f));
            ImGui::PopFont();
            // If hovered, and no admin rights while auto config is checked, and a connect method must be selected
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) &&
                (!m_Entry.ready(handles->devices, m_Entry) || !enableConnectButton) &&
                (connectMethodSelector == AUTO_CONNECT || connectMethodSelector == MANUAL_CONNECT))
            {
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLBlueIsh);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                ImGui::BeginTooltip();
                std::vector<std::string> errors = m_Entry.getNotReadyReasons(handles->devices, m_Entry);
                ImGui::Text("Please solve the following: ");
                if (connectMethodSelector == AUTO_CONNECT)
                {
                    if (reader)
                        errors.insert(errors.begin(), "No device selected");
                    else
                        errors.insert(errors.begin(), "Please Start AutoConnect");

                    Utils::removeFromVector(&errors, "No selected network adapter");
                }
                if (elevated() && connectMethodSelector == MANUAL_CONNECT && handles->configureNetwork)
                {
#ifdef WIN32
                    errors.emplace_back("Admin rights is needed to auto configure network");
#else
                    errors.emplace_back("Root access is needed to auto configure network");
#endif
                }
                for (size_t i = 0; i < errors.size(); ++i)
                {
                    ImGui::Text("%s", (std::to_string(i + 1) + ". " + errors[i]).c_str());
                }

                ImGui::EndTooltip();
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
            }
            if (!m_Entry.ready(handles->devices, m_Entry) || !enableConnectButton)
            {
                ImGui::PopStyleColor();
                ImGui::PopItemFlag();
            }
            ImGui::Dummy(ImVec2(0.0f, 5.0f));

            if (btnCancel)
            {
                handles->usageMonitor->userClickAction("Cancel", "button",
                                                       ImGui::GetCurrentWindow()->Name);
                ImGui::CloseCurrentPopup();
            }

            if (btnConnect && m_Entry.ready(handles->devices, m_Entry) && enableConnectButton)
            {
                handles->usageMonitor->userClickAction("Connect", "button",
                                                       ImGui::GetCurrentWindow()->Name);
                if (reader)
                {
                    reader->setIpConfig(resultsComboIndex);
                    reader->sendStopSignal();
                }
                // Stop autoConnect and set IP the found MultiSense device
                // Next: Create default element, but if this happens on windows we can to connect for ~~8 seconds to allow added ip address to finish configuring
#ifdef WIN32
                createDefaultElement(handles, m_Entry, connectMethodSelector == AUTO_CONNECT);
                ImGui::CloseCurrentPopup();
#else
                createDefaultElement(handles, m_Entry);
                ImGui::CloseCurrentPopup();
#endif
                entryConnectDeviceList.clear();
                resultsComboIndex = -1;
            }

            ImGui::EndPopup();
        }
        else
        {
            adapterUtils.stopAdapterScan(); // Stop scan if we close popup
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(5); // popup style vars
    }

    void addSpinnerGif(VkRender::GuiObjectHandles* handles)
    {
        ImVec2 size = ImVec2(40.0f, 40.0f);
        ImVec2 uv0 = ImVec2(0.0f, 0.0f);
        ImVec2 uv1 = ImVec2(1.0f, 1.0f);

        ImVec4 bg_col = ImVec4(0.054f, 0.137f, 0.231f, 1.0f);
        ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

        ImGui::Image(handles->info->gif.image[gifFrameIndex], size, uv0, uv1, bg_col, tint_col);
        auto time = std::chrono::steady_clock::now();
        auto time_span =
            std::chrono::duration_cast<std::chrono::duration<float>>(time - gifFrameTimer);

        if (time_span.count() > static_cast<float>(*handles->info->gif.delay) / 1000.0f)
        {
            gifFrameTimer = std::chrono::steady_clock::now();
            gifFrameIndex++;
        }

        if (gifFrameIndex == handles->info->gif.totalFrames)
            gifFrameIndex = 0;
    }
};


#endif //MULTISENSE_SIDEBAR_H
