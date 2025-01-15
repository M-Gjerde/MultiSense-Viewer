//
// Created by magnus on 10/2/23.
//

#include "Viewer/Rendering/ImGui/LayerFactory.h"
#include "Viewer/Rendering/ImGui/LayerExample.h"
#include "Viewer/Rendering/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/Rendering/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/Rendering/Editors/EditorUILayer.h"
#include "Viewer/Rendering/Editors/SceneHierarchy//SceneHierarchyLayer.h"
#include "Viewer/Rendering/Editors/Test/EditorTestLayer.h"
#include "Viewer/Rendering/Editors/MenuLayer.h"
#include "Viewer/Rendering/Editors/MainContextLayer.h"

#include "Viewer/Rendering/Editors/MultiSenseViewer/ConfigurationEditor/WelcomeScreenLayer.h"
#include "Viewer/Rendering/Editors/MultiSenseViewer/ConfigurationEditor/ConfigurationLayer.h"
#include "Viewer/Rendering/Editors/MultiSenseViewer/SidebarEditor/SideBarLayer.h"

#include "Viewer/Rendering/Editors/3DViewport/Editor3DLayer.h"
#include "Viewer/Rendering/Editors/Properties/PropertiesLayer.h"
#include "Viewer/Rendering/Editors/GaussianViewer/EditorGaussianViewerLayer.h"
#include "Viewer/Rendering/Editors/ImageEditor/EditorImageLayer.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayer.h"

namespace VkRender {


    std::shared_ptr<VkRender::Layer> LayerFactory::createLayer(const std::string &layerName) {

        if (layerName == "LayerExample") return std::make_shared<LayerExample>();
        if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
        if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
        if (layerName == "WelcomeScreenLayer") return std::make_shared<WelcomeScreenLayer>();
        if (layerName == "SideBarLayer") return std::make_shared<SideBarLayer>();
        if (layerName == "MenuLayer") return std::make_shared<MenuLayer>();
        if (layerName == "EditorUILayer") return std::make_shared<EditorUILayer>();
        if (layerName == "SceneHierarchyLayer") return std::make_shared<SceneHierarchyLayer>();
        if (layerName == "EditorTestLayer") return std::make_shared<EditorTestLayer>();
        if (layerName == "MainContextLayer") return std::make_shared<MainContextLayer>();
        if (layerName == "ConfigurationLayer") return std::make_shared<ConfigurationLayer>();
        if (layerName == "Editor3DLayer") return std::make_shared<Editor3DLayer>();
        if (layerName == "PropertiesLayer") return std::make_shared<PropertiesLayer>();
        if (layerName == "EditorGaussianViewerLayer") return std::make_shared<EditorGaussianViewerLayer>();
        if (layerName == "EditorImageLayer") return std::make_shared<EditorImageLayer>();
        if (layerName == "EditorPathTracerLayer") return std::make_shared<EditorPathTracerLayer>();

        throw std::runtime_error("Tried to push layer: " + layerName + " Which doesn't exists");
    }
};