/**
 * @file: MultiSense-Viewer/include/Viewer/Rendering/ImGui/Layer.h
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
#include <imgui.h>

#include "Viewer/Application/pch.h"
#include "Viewer/Application/UsageMonitor.h"

namespace VkRender {
    class Application;
    class Entity;
    class Scene;
    class Editor;
    /** @brief Set of Default colors */
    namespace Colors {
        static const ImVec4 green(0.26f, 0.42f, 0.31f, 1.0f);
        static const ImVec4 red(0.613f, 0.045f, 0.046f, 1.0f);
        static const ImVec4 CRLGray421(0.666f, 0.674f, 0.658f, 1.0f);
        static const ImVec4 CRLGray424(0.411f, 0.419f, 0.407f, 1.0f);
        static const ImVec4 CRLCoolGray(0.870f, 0.878f, 0.862f, 1.0f);
        static const ImVec4 CRLGray424Main(0.462f, 0.474f, 0.494f, 1.0f);
        static const ImVec4 CRLDarkGray425(0.301f, 0.313f, 0.309f, 1.0f);
        static const ImVec4 CRLFrameBG(0.196f, 0.2392f, 0.2588f, 1.0f);
        static const ImVec4 CRLRed(0.768f, 0.125f, 0.203f, 1.0f);
        static const ImVec4 CRLRedHover(0.86f, 0.378f, 0.407f, 1.0f);
        static const ImVec4 CRLRedActive(0.96f, 0.478f, 0.537f, 1.0f);
        static const ImVec4 CRLBlueIsh(0.313f, 0.415f, 0.474f, 1.0f);
        static const ImVec4 CRLTextGray(0.2f, 0.2f, 0.2f, 1.0f);
        static const ImVec4 CRLTextWhite(0.9f, 0.9f, 0.9f, 1.0f);
    }

    /** @brief block for simulated camera, Mostly used for testing  */
    struct CameraSelection {
        std::string tag = "Default Camera";
        bool enabled = false;
        bool selected = false;
        int currentItemSelected = 0;

        struct Info {
            /** @brief 3D view camera type for this device. Arcball or first person view controls) */
            int type = 0;
            /** @brief Reset 3D view camera position and rotation */
            bool reset = false;
        };

        std::unordered_map<std::string, Info> info;
    };

    /** @Brief Holds paths to various UI inpits stuff */
    struct Paths {
        std::filesystem::path loadColMapPosesPath;
        std::filesystem::path importFilePath;

        bool updateObjPath = false;
        bool update3DGSPath = false;
        bool updateGLTFPath = false;
    };

    /** @brief Handle which is the MAIN link between ''frontend and backend'' */
    struct GuiResourcesData {
        /** @brief GUI window info used for creation and updating */
        /**@brief Font types used throughout the gui. usage: ImGui::PushFont(font13).. Initialized in GuiManager class */
        ImFont *font8{}, *font13{}, *font15, *font18{}, *font24{}, *fontIcons{};

        /** @brief
        * Container to hold animated gif images
        */
        struct {
            ImTextureID image[20]{};
            uint32_t totalFrames{};
        } gif{};

        /** @brief Containing descriptor handles for each image button texture */
        std::vector<ImTextureID> imageButtonTextureDescriptor;
    };

    /**
     * @brief A UI Layer drawn by \refitem GuiManager.
     * To add an additional UI layer see \refitem LayerExample.
     */
    class Layer {
    public:

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
        virtual void onUIRender() = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called after draw command have been recorded, but before this frame has ended.
         * Can be used to prepare for next frame for instance
         */
        virtual void onFinishedRender() = 0;

        virtual void setScene(std::weak_ptr<Scene> scene){
            //m_scene = scene;
        }
        virtual void setContext(Application* ctx){
            m_context = ctx;
        }

        Application* m_context;
        Editor* m_editor;
    };
}

#endif //MULTISENSE_LAYER_H
