//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_RENDERER_H
#define MULTISENSE_RENDERER_H


// System
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <filesystem>

#include <MultiSense/src/core/VulkanRenderer.h>
#include <MultiSense/src/core/Base.h>
#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/tools/Macros.h>
#include <MultiSense/src/core/CameraConnection.h>

// Include scripts
//#include <MultiSense/src/scripts/pointcloud/VirtualPointCloud.h>
//#include <MultiSense/src/scripts/Example.h>
//#include <MultiSense/src/scripts/objects/LightSource.h>
#include <MultiSense/src/scripts/video/DefaultPreview.h>
#include <MultiSense/src/scripts/pointcloud/PointCloud.h>
#include <MultiSense/src/scripts/video/PreviewBar.h>


class Renderer : VulkanRenderer {

public:

    /**
     * @brief Default constructor for renderer
     * @param title Title of application
     */
    explicit Renderer(const std::string &title) : VulkanRenderer(title, true) {
        // During constructor prepare backend for rendering
        this->title = title;
        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;
        prepareRenderer();
    };
     ~Renderer() override = default;

    void render() override;
    void prepareRenderer();
    void run() {
        renderLoop();
    }

    void draw();

private:

    std::vector<std::unique_ptr<Base>> scripts;
    CameraConnection *cameraConnection;


protected:

    void updateUniformBuffers();

    void viewChanged() override;
    void addDeviceFeatures() override;
    void buildCommandBuffers() override;
    /**
     * Overrides UIUpdate function. Is called with an per-frame updated handle to \ref UISettings
     * @param uiSettings Handle to UISetting variables
     */
    void UIUpdate(GuiObjectHandles *uiSettings) override;

    /**
     *
     * \brief creates instances from classes located in src/scripts directory. Usually each class here represents object(s) in the scene
     */
    void generateScriptClasses();

};


#endif //AR_ENGINE_RENDERER_H
