/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/Layer.h
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

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#define IMGUI_INCLUDE_IMGUI_USER_H
#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DEFINE_MATH_OPERATORS
#include "Viewer/Tools/Macros.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
#include <imgui.h>
DISABLE_WARNING_POP
#include <array>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>

#include "Viewer/Renderer/UsageMonitor.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/KeyInput.h"
#include "Viewer/Core/MultiSenseDeviceDefinitions.h"
#include "Viewer/Tools/ThreadPool.h"

namespace VkRender {

    /** @brief Set of Default colors */
    namespace Colors {
        static const ImVec4 green(0.26f, 0.42f, 0.31f, 1.0f);
        static const ImVec4 TextGreenColor(0.16f, 0.95f, 0.11f, 1.0f);
        static const ImVec4 TextRedColor(0.95f, 0.345f, 0.341f, 1.0f);
        static const ImVec4 red(0.613f, 0.045f, 0.046f, 1.0f);
        static const ImVec4 DarkGray(0.1f, 0.1f, 0.1f, 1.0f);
        static const ImVec4 PopupTextInputBackground(0.01f, 0.05f, 0.1f, 1.0f);
        static const ImVec4 TextColorGray(0.75f, 0.75f, 0.75f, 1.0f);
        static const ImVec4 PopupHeaderBackgroundColor(0.15f, 0.25, 0.4f, 1.0f);
        static const ImVec4 PopupBackground(0.183f, 0.33f, 0.47f, 1.0f);
        static const ImVec4 CRLGray421(0.666f, 0.674f, 0.658f, 1.0f);
        static const ImVec4 CRLGray424(0.411f, 0.419f, 0.407f, 1.0f);
        static const ImVec4 CRLCoolGray(0.870f, 0.878f, 0.862f, 1.0f);
        static const ImVec4 CRLCoolGrayTransparent(0.870f, 0.878f, 0.862f, 0.5f);
        static const ImVec4 CRLGray424Main(0.462f, 0.474f, 0.494f, 1.0f);
        static const ImVec4 CRLDarkGray425(0.301f, 0.313f, 0.309f, 1.0f);
        static const ImVec4 CRLFrameBG(0.196f, 0.2392f, 0.2588f, 1.0f);
        static const ImVec4 CRLRed(0.768f, 0.125f, 0.203f, 1.0f);
        static const ImVec4 CRLRedTransparent(0.768f, 0.125f, 0.203f, 0.85f);
        static const ImVec4 CRLRedHover(0.86f, 0.378f, 0.407f, 1.0f);
        static const ImVec4 CRLRedActive(0.96f, 0.478f, 0.537f, 1.0f);
        static const ImVec4 CRLBlueIsh(0.313f, 0.415f, 0.474f, 1.0f);
        static const ImVec4 CRLBlueIshTransparent(0.313f, 0.415f, 0.474f, 0.3f);
        static const ImVec4 CRLBlueIshTransparent2(0.313f, 0.415f, 0.474f, 0.1f);
        static const ImVec4 CRLTextGray(0.2f, 0.2f, 0.2f, 1.0f);
        static const ImVec4 CRLTextLightGray(0.4f, 0.4f, 0.4f, 1.0f);
        static const ImVec4 CRLTextWhite(0.9f, 0.9f, 0.9f, 1.0f);
        static const ImVec4 CRLTextWhiteDisabled(0.75f, 0.75f, 0.75f, 1.0f);
        static const ImVec4 CRL3DBackground(0.0f, 0.0f, 0.0f, 1.0f);
    }

    struct GuiLayerUpdateInfo {
        bool firstFrame{};
        /** @brief Width of window surface */
        float width{};
        /** @brief Height of window surface */
        float height{};
        /** @brief aspect ratio of window surface */
        float aspect{};
        /**@brief Width of sidebar*/
        float sidebarWidth = 200.0f;
        /**@brief Size of elements in sidebar */
        float elementHeight = 140.0f;
        /**@brief Width of sidebar*/
        float debuggerWidth = 960.0f * 0.75f;
        /**@brief Size of elements in sidebar */
        float debuggerHeight = 480.0f * 0.75f;
        /**@brief Width of sidebar*/
        float metricsWidth = 350.0f;
        /**@brief Physical Graphics m_Device used*/
        std::string deviceName = "DeviceName";
        /**@brief Title of Application */
        std::string title = "TitleName";
        /**@brief array containing the last 50 entries of frametimes */
        std::array<float, 50> frameTimes{};
        /**@brief min/max values for frametimes, used for updating graph*/
        float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
        /**@brief value for current frame timer*/
        float frameTimer{};
        /** @brief Current frame*/
        uint64_t frameID = 0;
        /**@brief Font types used throughout the gui. usage: ImGui::PushFont(font13).. Initialized in GuiManager class */
        ImFont *font8{}, *font13{}, *font15, *font18{}, *font24{};

        /** @brief Containing descriptor handles for each image button texture */
        std::vector<ImTextureID> imageButtonTextureDescriptor;

        /** @brief
        * Container to hold animated gif images
        */
        struct {
            ImTextureID image[20]{};
            uint32_t id{};
            uint32_t lastFrame = 0;
            uint32_t width{};
            uint32_t height{};
            uint32_t imageSize{};
            uint32_t totalFrames{};
            uint32_t *delay{};
        } gif{};

        /** @brief ImGUI Overlay on previews window sizes */
        float viewAreaElementSizeY = {0};
        float viewAreaElementSizeX = {0};
        float previewBorderPadding = 60.0f;
        /** @brief add m_Device button params */
        float addDeviceBottomPadding = 45.0f;
        float addDeviceWidth = 180.0f, addDeviceHeight = 35.0f;
        /** @brief Height of popupModal*/
        float popupHeight = 350.0f;
        /** @brief Width of popupModal*/
        float popupWidth = 550.0f;
        /** @brief Width of popupModal*/
        float metadataWidth = 700.0f;
        /** @brief Width of popupModal*/
        float metadataHeight = 600.0f;
        /**@brief size of control area tabs*/
        float tabAreaHeight = 60.0f;
        /** @brief size of Control Area*/
        float controlAreaWidth = 440.0f, controlAreaHeight = height;
        int numControlTabs = 2;
        /** @brief size of viewing Area*/
        float viewingAreaWidth = width - controlAreaWidth - sidebarWidth, viewingAreaHeight = height;

        ImVec2 viewingAreaWindowPos;
        /** @brief Horizontal size of scrollbar */
        float scrollbarSize = 10.0f;
        bool hoverState = false;
        bool isViewingAreaHovered = false;
        bool is3DTopBarHovered = false;

    };

    struct CameraUIBlock {
        /** @brief 3D view camera type for this device. Arcball or first person view controls) */
        int type = 0;
        /** @brief Reset 3D view camera position and rotation */
        bool reset = false;

        glm::vec3 pos;
        glm::vec3 up;
        glm::vec3 target;
        glm::vec3 cameraFront;

    };

    /** @brief block for simulated camera, Mostly used for testing  */
    struct CameraSimulator {
        bool enabled = false;

    };

    /** @brief Handle which is the MAIN link between ''frontend and backend'' */
    struct GuiObjectHandles {
        /** @brief Handle for current devices located in sidebar */
        std::vector<Device> devices;
        /** @brief GUI window info used for creation and updating */
        std::unique_ptr<GuiLayerUpdateInfo> info{};
        /** User action to configure network automatically even when using manual approach **/
        bool configureNetwork = false;
        /** Keypress and mouse events */
        float accumulatedActiveScroll = 0.0f;
        /**  Measures change in scroll. Does not provide change the accumulated scroll is outside of min and max scroll  */
        float scroll = 0.0f;
        std::unordered_map<StreamWindowIndex, float> previewZoom{};
        float minZoom = 1.0f;
        float maxZoom = 4.5f;
        /** @brief min/max scroll used in DoublePreview layout */
        float maxScroll = 450.0f, minScroll = -550.0f;
        /** @brief Input from backend to IMGUI */
        const Input *input{};
        std::array<float, 4> clearColor{};
        /** @brief Display the debug window */
        bool showDebugWindow = false;

        /** @brief Open the popup window */
        bool openAddDevicePopup = false;
        bool renderer3D = false;


        std::shared_ptr<UsageMonitor> usageMonitor;
        /** @brief if a new version has been launched by crl*/
        bool newVersionAvailable = false;
        bool askUserForNewVersion = true;

        const VkRender::MouseButtons *mouse;
        VkRender::RenderBlock renderBlock;
        /** @brief Initialize \refitem clearColor because MSVC does not allow initializer list for std::array */
        GuiObjectHandles() {
            clearColor[0] = 0.870f;
            clearColor[1] = 0.878f;
            clearColor[2] = 0.862f;
            clearColor[3] = 1.0f;

            // Initialize map used for zoom for each preview window
            previewZoom[CRL_PREVIEW_ONE] = 1.0f;
            previewZoom[CRL_PREVIEW_TWO] = 1.0f;
            previewZoom[CRL_PREVIEW_THREE] = 1.0f;
            previewZoom[CRL_PREVIEW_FOUR] = 1.0f;
        }

        /** @brief Reference to threadpool held by GuiManager */
        std::shared_ptr<ThreadPool> pool{};

        CameraUIBlock camera{};
        CameraSimulator simulator{}; // TODO REMOVE
    };

    /**
     * @brief A UI Layer drawn by \refitem GuiManager.
     * To add an additional UI layer see \refitem LayerExample.
     */
    class Layer {

    public:

        virtual ~Layer() = default;

        /** @brief
         * Pure virtual must be overridden.
         * Called ONCE after UI have been initialized
         */
        virtual void onAttach() = 0;

        /** @brief
         * Pure virtual must be overridden.
         * Called ONCE before UI objects are destroyed
         */
        virtual void onDetach() = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called per frame, but before each script (\refitem Example) is updated
         * @param handles a UI object handle to collect user input
         */
        virtual void onUIRender(GuiObjectHandles *handles) = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called after draw command have been recorded, but before this frame has ended.
         * Can be used to prepare for next frame for instance
         */
        virtual void onFinishedRender() = 0;
    };


    /** @brief An initialized object needed to create a \refitem Device */
    struct EntryConnectDevice {
        std::string profileName = "MultiSense";
        std::string IP = "10.66.171.21";
        std::string interfaceName;
        std::string description;
        uint32_t interfaceIndex{};

        std::string cameraName;
        bool isRemoteHead = false;
        EntryConnectDevice() = default;

        EntryConnectDevice(std::string ip, std::string iName, std::string camera, uint32_t idx, std::string desc) : IP(
                std::move(ip)),
                                                                                                                    interfaceName(
                                                                                                                            std::move(
                                                                                                                                    iName)),
                                                                                                                    description(
                                                                                                                            std::move(
                                                                                                                                    desc)),
                                                                                                                    interfaceIndex(
                                                                                                                            idx),
                                                                                                                    cameraName(
                                                                                                                            std::move(
                                                                                                                                    camera)) {
            profileName.reserve(64);
            IP.reserve(16);
        }

        void reset() {
            profileName = "";
            IP = "";
            interfaceName = "";
            interfaceIndex = 0;
        }

        /**
         * @brief Utility function to check if the requested profile in \ref m_Entry is not conflicting with any of the previously connected devices in the sidebar
         * @param devices list of current devices
         * @param entry new connection to be added
         * @return true of we can add this new profile to list. False if not
         */
        bool ready(const std::vector<VkRender::Device> &devices, const EntryConnectDevice &entry) const {
            bool profileNameEmpty = entry.profileName.empty();
            bool profileNameTaken = false;
            bool IPEmpty = entry.IP.empty();
            bool adapterNameEmpty = entry.interfaceName.empty();

            bool AdapterAndIPInTaken = false;
            // Loop through devices and check that it doesn't exist already.
            for (auto &d: devices) {
                if (d.IP == entry.IP && d.interfaceName == entry.interfaceName) {
                    //AdapterAndIPInTaken = true;
                    //Log::Logger::getInstance()->info("Ip {} on adapter {} already in use", entry.IP, entry.interfaceName);
                }
                if (d.name == entry.profileName) {
                    profileNameTaken = true;
                    //Log::Logger::getInstance()->info("Profile m_Name '{}' already taken", entry.profileName);
                }

            }
            bool ready = true;
            if (profileNameEmpty || profileNameTaken || IPEmpty || adapterNameEmpty || AdapterAndIPInTaken)
                ready = false;
            return ready;
        }

        std::vector<std::string> getNotReadyReasons(const std::vector<VkRender::Device> &devices, const EntryConnectDevice &entry){
            std::vector<std::string> errors;
            bool profileNameEmpty = entry.profileName.empty();
            bool IPEmpty = entry.IP.empty();
            bool adapterNameEmpty = entry.interfaceName.empty();
            // Loop through devices and check that it doesn't exist already.
            for (auto &d: devices) {
                if (d.IP == entry.IP && d.interfaceName == entry.interfaceName) {
                    //errors.emplace_back("The IP address on the selected adapter is in use");
                }
                if (d.name == entry.profileName) {
                    errors.emplace_back("Profile name already in use");
                }

            }
            if (profileNameEmpty)
                errors.emplace_back("Profile name cannot be left blank");

            if (IPEmpty)
                errors.emplace_back("IP Address cannot be left blank");
            if (adapterNameEmpty)
                errors.emplace_back("No selected network adapter");

            return errors;

        }
    };

}

#endif //MULTISENSE_LAYER_H
