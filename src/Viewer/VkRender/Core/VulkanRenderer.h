/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Core/VulkanRenderer.h
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
 *   2021-09-4, mgjerde@carnegierobotics.com, Created file.
 **/


#ifndef MULTISENSE_VULKANRENDERER_H
#define MULTISENSE_VULKANRENDERER_H

#include <string>
#include <vector>
#include <vector>
#include <cstring>
#include <iostream>
#include <chrono>

#include "Viewer/Tools/Macros.h"
DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
#include <vk_mem_alloc.h>
DISABLE_WARNING_POP

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#ifdef APIENTRY
    #undef APIENTRY
#endif


#include "Viewer/VkRender/ImGui/GuiManager.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/Core/VulkanSwapchain.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"


namespace VkRender {

    class VulkanRenderer {

    public:
        explicit VulkanRenderer(const std::string &title);

        virtual ~VulkanRenderer();

        /** @brief Setup the vulkan instance, flashing required extensions and connect to the physical m_Device (GPU) */
        bool initVulkan();

        /** @brief Example settings that can be changed by ... */
        struct Settings {
            /** @brief Activates validation layers (and message output) when set to true */
            bool validation = false;
            /** @brief Set to true if fullscreen mode has been requested via command line */
            bool fullscreen = false;
            /** @brief Set to true if v-sync will be forced for the swapchain */
            bool vsync = false;
            /** @brief Enable UI overlay */
            bool overlay = true;
        } m_settings;

        struct GLFWCursors {
            GLFWcursor* arrow = nullptr;
            GLFWcursor* resizeVertical = nullptr;
        }m_cursors;


        std::string m_title = "-1";
        std::string m_name = "VulkanRenderer";
        /** @brief This application is written against Vulkan API v.1.1+ **/
        uint32_t apiVersion = VK_API_VERSION_1_1;
        bool backendInitialized = false;
        uint32_t m_width = 1920;      // Default values - Actual values set in constructor
        uint32_t m_height = 1080;     // Default values - Actual values set in constructor

        /** @brief Encapsulated physical and logical vulkan m_Device */
        VulkanDevice* m_vulkanDevice{};


        /** @brief Last frame time measured using a high performance timer (if available) */
        float frameTimer = 1.0f;
        std::chrono::system_clock::time_point rendererStartTime;
        float runTime = 0.0f;

        glm::vec2 mousePos{};
        VkSampleCountFlagBits msaaSamples{};

        VkRender::MouseButtons mouseButtons{};
        float mouseScrollSpeed = 0.1f;

        int keyPress = -1;
        int keyAction = -1;
        Input input;

        /** @brief Handle for Logging*/
        Log::Logger *m_logger = nullptr; // Create the object pointer for Logger Class

        /** @brief Handle for UI updates and overlay */

        /** @brief (Virtual) Creates the application wide Vulkan instance */
        virtual VkResult createInstance(bool enableValidation);

        /** @brief (Pure virtual) Render function to be implemented by the application */
        virtual void recordCommands() = 0;

        /** @brief (Pure virtual) compute render function to be implemented by the application */
        virtual bool compute() = 0;

        /** @brief (Pure virtual) compute render function to be implemented by the application */
        virtual void updateUniformBuffers() = 0;

        /** @brief (Virtual) Called when the camera m_View has changed */
        virtual void viewChanged();

        /** @brief (Virtual) Called after the mouse cursor moved and before internal events (like camera m_Rotation) is firstUpdate */
        virtual void mouseMoved(float x, float y, bool &handled);
        /** @brief (Virtual) Called after the mouse cursor moved and before internal events (like camera m_Rotation) is firstUpdate */
        virtual void keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) = 0;

        /** @brief (Virtual) Called after the mouse cursor moved and before internal events (like camera m_Rotation) is firstUpdate */
        virtual void mouseScroll(float change);

        /** @brief (Virtual) Called when the window has been resized, can be used by the sample application to recreate resources */
        virtual void windowResized();

        virtual void postRenderActions() = 0; // TODO test

        /** @brief (Virtual) Called when resources have been recreated that require a rebuild of the command buffers (e.g. frame buffer), to be implemented by the sample application */
        virtual void buildCommandBuffers();

        /** @brief (Virtual) Called after the physical m_Device m_Features have been read, can be used to set m_Features to flashing on the m_Device */
        virtual void addDeviceFeatures() = 0;

        /** @brief Prepares all Vulkan resources and functions required to run the sample */
        virtual void prepare();

        /** @brief Entry point for the main render loop */
        void renderLoop();

        /** Prepare the next frame for workload submission by acquiring the next swap chain m_Image */
        void prepareFrame();

        /** @brief Presents the current m_Image to the swap chain */
        void submitFrame();

        virtual void closeApplication();


    protected:
        // Window instance GLFW
        GLFWwindow *window;
        // Vulkan Instance, stores al per-application states
        VkInstance instance{};
        // Physical Device that Vulkan will use
        VkPhysicalDevice physicalDevice{};
        //Physical m_Device m_Properties (m_Device limits etc..)
        VkPhysicalDeviceProperties deviceProperties{};
        // Features available on the physical m_Device
        VkPhysicalDeviceFeatures deviceFeatures{};
        /** @brief Set of physical m_Device m_Features to be enabled for this example (must be set in the derived constructor) */
        VkPhysicalDeviceFeatures enabledFeatures{};
        // Features all available memory types for the physical m_Device
        VkPhysicalDeviceMemoryProperties deviceMemoryProperties{};
        /** @brief Set of m_Device extensions to be enabled for this example (must be set in the derived constructor) */

        std::vector<const char *> enabledDeviceExtensions;
        std::vector<const char *> enabledInstanceExtensions;
        /** @brief vma allocator */
        VmaAllocator m_allocator{};

        /** @brief Logical m_Device, application's m_View of the physical m_Device (GPU) */
        VkDevice device{};
        // Handle to the m_Device graphics queue that command buffers are submitted to
        VkQueue graphicsQueue{};
        // Handle to the m_Device compute queue that command buffers are submitted to
        VkQueue computeQueue{};
        /**@brief synchronozation for vkQueueSubmit*/
        std::mutex queueSubmitMutex;
        // Depth buffer m_Format (selected during Vulkan initialization)
        VkFormat depthFormat{};
        // Wraps the swap chain to present images (framebuffers) to the windowing system
        std::unique_ptr<VulkanSwapchain> swapchain;
        // Synchronization semaphores
        struct Semaphores{
            // Swap chain m_Image presentation
            VkSemaphore presentComplete;
            // Command buffer submission and execution
            VkSemaphore renderComplete;
            // compute submission
            VkSemaphore computeComplete;
            // Cuda semaphore handle
        };
        std::vector<Semaphores> semaphores;

        std::vector<VkFence> waitFences{};
        std::vector<VkFence> computeInFlightFences{};
        // if we have work in our compute command buffers
        std::vector<bool> hasWorkFromLastSubmit{};

        /** @brief Pipeline stages used to wait at for graphics queue submissions */
        VkPipelineStageFlags submitPipelineStages[2] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        // Contains command buffers and semaphores to be presented to the queue
        VkSubmitInfo submitInfo{};
        // CommandPool for render command buffers
        VkCommandPool cmdPool{};
        // CommandPool for compute command buffers
        VkCommandPool cmdPoolCompute{};
        // Command buffers used for rendering
        CommandBuffer drawCmdBuffers{};
        // Command buffers used for compute
        CommandBuffer computeCommand{};
        // Global render pass for frame buffer writes
        //VkRenderPass renderPass{};
        // List of available frame buffers (same as number of swap chain images)
        std::vector<VkFramebuffer> frameBuffers{};
        // Active frame buffer index
        uint32_t currentFrame = 0;
        // Active image index in swapchain
        uint32_t imageIndex = 0;
        // Pipeline cache object
        VkPipelineCache pipelineCache{};
        // Handle to Debug Utils
        VkDebugUtilsMessengerEXT debugUtilsMessenger{};

        int frameCounter = 0;
        int frameID = 0;
        bool recreateResourcesNextFrame = false;
        void setMultiSampling(VkSampleCountFlagBits samples);

    private:
        uint32_t destWidth{};
        uint32_t destHeight{};
        float lastFPS{};

        VkSampleCountFlagBits getMaxUsableSampleCount();

        void windowResize();

        static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

        static void resizeCallback(GLFWwindow *window, int width, int height);

        static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos);

        static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);

        static void charCallback(GLFWwindow *window, unsigned int codepoint);

        static void mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset);

        void createCommandPool();

        void createCommandBuffers();

        void createSynchronizationPrimitives();

        void createPipelineCache();

        void destroyCommandBuffers();

        void setWindowSize(uint32_t width, uint32_t height);

        /** @brief Default function to handle cursor m_Position input, calls the override function mouseMoved(...) **/
        void handleMouseMove(float x, float y);

        [[nodiscard]] VkPhysicalDevice pickPhysicalDevice(std::vector<VkPhysicalDevice> devices) const;

        static int ImGui_ImplGlfw_TranslateUntranslatedKey(int key, int scancode);

        static ImGuiKey ImGui_ImplGlfw_KeyToImGuiKey(int key);

#ifdef WIN32
        void clipboard();
#endif

        void createColorResources();

        void computePipeline();

        static bool checkInstanceExtensionSupport(const std::vector<const char *> &checkExtensions);

    };
}
#endif //MULTISENSE_VULKANRENDERER_H
