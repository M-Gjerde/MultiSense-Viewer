//
// Created by magnus on 10/2/23.
//

#define IMGUI_INCLUDE_IMGUI_USER_H
#include "Viewer/ImGui/LayerFactory.h"

#include "Viewer/ImGui/SideBarLayer.h"
#include "Viewer/ImGui/WelcomeScreenLayer.h"
#include "Viewer/ImGui/MainLayer.h"
#include "Viewer/ImGui/LayerExample.h"
#include "Viewer/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/ImGui/AdditionalWindows/CustomMetadata.h"
#include "Viewer/ImGui/Renderer3D/Renderer3DLeft.h"
#include "Viewer/ImGui/Renderer3D/Renderer3DBottom.h"
#include "Viewer/ImGui/Renderer3D/Renderer3DRight.h"

std::shared_ptr<VkRender::Layer> LayerFactory::createLayer(const std::string& layerName) {
    if (layerName == "SideBarLayer") return std::make_shared<SideBarLayer>();
    if (layerName == "WelcomeScreenLayer") return std::make_shared<WelcomeScreenLayer>();
    if (layerName == "MainLayer") return std::make_shared<MainLayer>();
    if (layerName == "LayerExample") return std::make_shared<LayerExample>();
    if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
    if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
    if (layerName == "CustomMetadata") return std::make_shared<CustomMetadata>();
    if (layerName == "Renderer3DLeft") return std::make_shared<Renderer3DLeft>();
    if (layerName == "Renderer3DBottom") return std::make_shared<Renderer3DBottom>();
    if (layerName == "Renderer3DRight") return std::make_shared<Renderer3DRight>();

    return nullptr; // or throw an exception if an unknown layer is requested
}