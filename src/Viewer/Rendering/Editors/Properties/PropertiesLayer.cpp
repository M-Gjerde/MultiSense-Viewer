//
// Created by magnus on 8/23/24.
//

#include <glm/gtc/type_ptr.hpp>  // for glm::value_ptr

#include "Viewer/Rendering/Editors/Properties/PropertiesLayer.h"

#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"

#include "Viewer/Rendering/ImGui/Layer.h"

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

namespace VkRender {
    /** Called once upon this object creation**/
    void PropertiesLayer::onAttach() {
    }

    /** Called after frame has finished rendered **/
    void PropertiesLayer::onFinishedRender() {
    }

    void PropertiesLayer::setScene(std::weak_ptr<Scene> scene) {
        Layer::setScene(scene);
        m_selectionContext = Entity(); // reset selectioncontext
    }

    bool PropertiesLayer::drawVec3Control(const std::string& label, glm::vec3& values, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO& io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize)) {
            values.x = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &values.x, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Y", buttonSize))
            values.y = resetValue;
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##Y", &values.y, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Z", buttonSize)) {
            values.z = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##Z", &values.z, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();

        return valueChanged;
    }

    bool PropertiesLayer::drawVec2Control(const std::string& label, glm::vec2& values, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO& io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize)) {
            values.x = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &values.x, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Y", buttonSize))
            values.y = resetValue;
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##Y", &values.y, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();


        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();

        return valueChanged;
    }

    bool PropertiesLayer::drawFloatControl(const std::string& label, float& value, float resetValue = 0.0f,
                                           float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO& io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.5f, 0.5f, 0.5f, 1.0f}); // Gray
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.6f, 0.6f, 0.6f, 1.0f}); // Lighter gray
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.4f, 0.4f, 0.4f, 1.0f}); // Darker gray
        ImGui::PushFont(boldFont);
        if (ImGui::Button("R", buttonSize)) {
            value = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &value, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();
        return valueChanged;
    }

    template <typename T, typename UIFunction>
    void PropertiesLayer::drawComponent(const std::string& componentName, Entity entity, UIFunction uiFunction) {
        const ImGuiTreeNodeFlags treeNodeFlags =
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth |
            ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap;
        if (entity.hasComponent<T>()) {
            auto& component = entity.getComponent<T>();
            ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

            float fontSize = ImGui::GetFontSize();
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});

            float lineHeight = fontSize + 4.0f * 2.0f;
            ImGui::Separator();
            bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, "%s", componentName.c_str());
            ImGui::PopStyleVar();
            ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);

            if (ImGui::Button("+", ImVec2{lineHeight, lineHeight})) {
                ImGui::OpenPopup("ComponentSettings");
            }

            bool removeComponent = false;
            if (ImGui::BeginPopup("ComponentSettings")) {
                if (componentName != "Tag") {
                    if (ImGui::MenuItem("Remove component"))
                        removeComponent = true;
                }
                ImGui::EndPopup();
            }

            if (open) {
                uiFunction(component);
                ImGui::TreePop();
            }

            if (removeComponent)
                entity.removeComponent<T>();
        }
    }

    void PropertiesLayer::drawComponents(Entity entity) {
        if (ImGui::Button("Add Component"))
            ImGui::OpenPopup("AddComponent");

        if (ImGui::BeginPopup("AddComponent")) {
            displayAddComponentEntry<CameraComponent>("Camera");
            displayAddComponentEntry<TransformComponent>("Transform");
            displayAddComponentEntry<MeshComponent>("Mesh");
            displayAddComponentEntry<MaterialComponent>("Material");
            displayAddComponentEntry<PointCloudComponent>("PointCloud");
            displayAddComponentEntry<GaussianComponent>("3DGS Model");
            displayAddComponentEntry<GaussianComponent2DGS>("2DGS Model");

            ImGui::EndPopup();
        }

        drawComponent<TransformComponent>("Transform", entity, [](TransformComponent& component) {
            bool paramsChanged = false;
            component.setMoving(paramsChanged);

            paramsChanged |= drawVec3Control("Translation", component.getPosition());
            paramsChanged |= drawVec3Control("Rotation", component.getRotationEuler(), 0.0f, 2.0f);
            paramsChanged |= drawVec3Control("Scale", component.getScale(), 1.0f);


            if (paramsChanged) {
                component.setMoving(paramsChanged);
                component.updateFromEulerRotation();
            }
        });
        drawComponent<CameraComponent>("Camera", entity, [this, entity](CameraComponent& component) {
            //drawFloatControl("Field of View", component.camera->fov(), 1.0f);
            bool paramsChanged = false;

            paramsChanged |= ImGui::Checkbox("Render scene from viewpoint", &component.isActiveCamera());
            if (paramsChanged && component.isActiveCamera()) {
                // The user has just activated this camera.
                // Iterate over all camera components in the scene.
                auto view = m_context->activeScene()->getRegistry().view<CameraComponent>();
                for (auto entityID : view) {
                    Entity localEntity(entityID, m_context->activeScene().get());
                    // Skip the current entity (the one the user just toggled)
                    if (localEntity == entity)
                        continue;
                    auto& otherCameraComponent = localEntity.getComponent<CameraComponent>();
                    // Deactivate any camera that is not the current one.
                    if (otherCameraComponent.isActiveCamera()) {
                        otherCameraComponent.isActiveCamera() = false;
                    }
                }
            }

            paramsChanged |= ImGui::Checkbox("Flip Y", &component.cameraSettings.flipY);
            ImGui::SameLine();
            paramsChanged |= ImGui::Checkbox("Flip X", &component.cameraSettings.flipX);

            static const auto allCameraTypes = CameraComponent::getAllCameraTypes();
            static const auto cameraTypeStrings = []() {
                std::vector<std::string> strings;
                for (const auto& type : allCameraTypes) {
                    strings.push_back(CameraComponent::cameraTypeToString(type));
                }
                return strings;
            }();

            // Get the current camera type as a string
            std::string currentCameraTypeStr = CameraComponent::cameraTypeToString(component.cameraType);

            // Get the index of the current camera type
            int currentIndex = std::distance(
                cameraTypeStrings.begin(),
                std::find(cameraTypeStrings.begin(), cameraTypeStrings.end(), currentCameraTypeStr)
            );

            // ImGui combo box
            if (ImGui::BeginCombo("Camera Type", currentCameraTypeStr.c_str())) {
                for (int i = 0; i < cameraTypeStrings.size(); ++i) {
                    bool isSelected = (i == currentIndex);
                    if (ImGui::Selectable(cameraTypeStrings[i].c_str(), isSelected)) {
                        currentIndex = i;
                        component.cameraType = allCameraTypes[i];
                        paramsChanged |= true;
                    }

                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }


            switch (component.cameraType) {
            case CameraComponent::PERSPECTIVE:
                paramsChanged |= ImGui::SliderFloat("Field of View", &component.baseCameraParameters.fov, 5.0f,
                                                    180.0f);
                paramsChanged |= ImGui::SliderFloat("Aspect Ratio", &component.baseCameraParameters.aspect, 0.1f,
                                                    10.0f);
                paramsChanged |= ImGui::SliderFloat("Near Plane", &component.baseCameraParameters.near, 0.01f,
                                                    10.0f);
                paramsChanged |= ImGui::SliderFloat("Far Plane", &component.baseCameraParameters.far, 1.0f,
                                                    1000.0f);

                break;
            case CameraComponent::PINHOLE:
                paramsChanged |= ImGui::SliderFloat("Width", &component.pinholeParameters.width, 1.0, 4096, "%.0f");
                paramsChanged |= ImGui::SliderFloat("Height", &component.pinholeParameters.height, 1.0f, 4096.0f,
                                                    "%.0f");

                paramsChanged |= ImGui::SliderFloat("Fx", &component.pinholeParameters.fx, 1.0f, 4096.0f);
                paramsChanged |= ImGui::SliderFloat("Fy", &component.pinholeParameters.fy, 1.0f, 4096.0f);
                paramsChanged |= ImGui::SliderFloat("Cx", &component.pinholeParameters.cx, 1.0f, 4096.0f);
                paramsChanged |= ImGui::SliderFloat("Cy", &component.pinholeParameters.cy, 1.0f, 4096.0f);
                paramsChanged |= ImGui::SliderFloat("Focal Length", &component.pinholeParameters.focalLength, 1.0f,
                                                    100.0f);
                paramsChanged |= ImGui::SliderFloat("Aperture", &component.pinholeParameters.fNumber, 0.0f, 32.0f);

                if (ImGui::Button("Set from viewport")) {
                    auto& ci = m_context->getViewport()->getCreateInfo();
                    component.pinholeParameters.width = ci.width;
                    component.pinholeParameters.height = ci.height;
                    component.pinholeParameters.cx = ci.width / 2;
                    component.pinholeParameters.cy = ci.height / 2;
                    paramsChanged = true;
                }
                break;
            case CameraComponent::ARCBALL:
                break;
            }

            component.resetUpdateState();
            // TODO not a nice way of ensuring single frame updates from camera properties. Can we make this less error-prone? Relying on paramsChanged variable and  m_updateTrigger in cameraComponent struct
            if (paramsChanged) {
                component.updateParametersChanged();
            }
            component.camera->updateProjectionMatrix();
        });

        drawComponent<MeshComponent>("Mesh", entity, [this, &entity](MeshComponent& component) {
            // Mesh Type Selection
            int currentMeshType = component.meshDataType();
            // Create the combo and check for interaction
            // Polygon Mode Control
            ImGui::Text("Polygon Mode:");
            const char* polygonModes[] = {"Line", "Fill"};
            int currentMode = (component.polygonMode() == VK_POLYGON_MODE_LINE) ? 0 : 1;
            if (ImGui::Combo("Polygon Mode", &currentMode, polygonModes, IM_ARRAYSIZE(polygonModes))) {
                if (currentMode == 0) {
                    component.polygonMode() = VK_POLYGON_MODE_LINE;
                }
                else {
                    component.polygonMode() = VK_POLYGON_MODE_FILL;
                }
                m_context->activeScene()->onComponentUpdated(entity, component);
            }
            component.updateMeshData = false;

            // Begin the combo box
            if (ImGui::BeginCombo("Mesh Type",
                                  meshDataTypeToString(static_cast<MeshDataType>(currentMeshType)).c_str())) {
                // Loop through the available mesh types
                for (size_t i = 0; i < meshDataTypeToArray().size(); ++i) {
                    bool isSelected = (currentMeshType == static_cast<int>(meshDataTypeToArray()[i]));
                    if (ImGui::Selectable(meshDataTypeToString(meshDataTypeToArray()[i]).c_str(), isSelected)) {
                        currentMeshType = static_cast<int>(meshDataTypeToArray()[i]);
                        component.meshDataType() = static_cast<MeshDataType>(currentMeshType);
                        component.updateMeshData = true;

                        // Trigger behavior when a new type is selected
                        switch (component.meshDataType()) {
                        case MeshDataType::OBJ_FILE:

                            EditorUtils::openImportFileDialog("Wavefront", {".obj"}, LayerUtils::OBJ_FILE,
                                                              &m_loadFileFuture);
                            break;

                        case MeshDataType::PLY_FILE:

                            EditorUtils::openImportFileDialog("Stanford .PLY", {".ply"}, LayerUtils::PLY_MESH,
                                                              &m_loadFileFuture);
                            break;
                        case MeshDataType::CYLINDER:
                            component.meshParameters = std::make_shared<CylinderMeshParameters>();
                            break;
                        case MeshDataType::CAMERA_GIZMO_PINHOLE:
                            component.meshParameters = std::make_shared<CameraGizmoPinholeMeshParameters>();
                            break;
                        case MeshDataType::CAMERA_GIZMO_PERSPECTIVE:
                            component.meshParameters = std::make_shared<CameraGizmoPerspectiveMeshParameters>();
                            break;
                        default:
                            Log::Logger::getInstance()->error("Unknown mesh type!");
                            break;
                        }
                    }

                    // Ensure the currently selected item remains selected
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }

                ImGui::EndCombo();
            }

            // Display different input fields based on mesh type
            switch (component.meshDataType()) {
            case OBJ_FILE: {
                auto params = std::dynamic_pointer_cast<OBJFileMeshParameters>(component.meshParameters);
                if (params) {
                    ImGui::Text("Mesh File:");
                    ImGui::Text("%s", params->path.empty() ? "" : params->path.c_str());
                }
            }
            break;
            case PLY_FILE: {
                auto params = std::dynamic_pointer_cast<PLYFileMeshParameters>(component.meshParameters);
                if (params) {
                    ImGui::Text("Mesh File:");
                    ImGui::Text("%s", params->path.empty() ? "" : params->path.c_str());
                }
            }
            break;

            case MeshDataType::CYLINDER: {
                auto cylinderParams = std::dynamic_pointer_cast<CylinderMeshParameters>(component.meshParameters);
                if (cylinderParams) {
                    bool paramsChanged = false;
                    paramsChanged |= drawVec3Control("Origin", cylinderParams->origin);
                    paramsChanged |= drawVec3Control("Direction", cylinderParams->direction);
                    paramsChanged |= ImGui::SliderFloat("Magnitude", &cylinderParams->magnitude, 0.0f, 100.0f);
                    paramsChanged |= ImGui::SliderFloat("Radius", &cylinderParams->radius, 0.001f, 0.1f);
                    if (paramsChanged) {
                        component.updateMeshData = true;
                    }
                }
                break;
            }
                ImGui::Dummy(ImVec2(5.0f, 5.0f));

            case MeshDataType::CAMERA_GIZMO_PINHOLE: {
                if (entity.hasComponent<CameraComponent>()) {
                    auto cameraGizmoParams = std::dynamic_pointer_cast<CameraGizmoPinholeMeshParameters>(
                        component.meshParameters);
                    if (cameraGizmoParams) {
                        auto& cameraParams = entity.getComponent<CameraComponent>().pinholeParameters;
                        // Update focal point and check if it has changed
                        if (cameraGizmoParams->parameters != cameraParams) {
                            cameraGizmoParams->parameters = cameraParams;
                            component.updateMeshData = true;
                        }
                    }
                }
                else {
                    if (ImGui::Button("Add a camera component!")) {
                        entity.addComponent<CameraComponent>();
                    }
                }
                break;

            case MeshDataType::CAMERA_GIZMO_PERSPECTIVE: {
                if (entity.hasComponent<CameraComponent>()) {
                    auto cameraGizmoParams = std::dynamic_pointer_cast<CameraGizmoPerspectiveMeshParameters>(
                        component.meshParameters);
                    if (cameraGizmoParams) {
                        auto& cameraParams = entity.getComponent<CameraComponent>().baseCameraParameters;
                        // Update focal point and check if it has changed
                        if (cameraGizmoParams->parameters != cameraParams) {
                            cameraGizmoParams->parameters = cameraParams;
                            component.updateMeshData = true;
                        }
                    }
                }
                else {
                    if (ImGui::Button("Add a camera component!")) {
                        entity.addComponent<CameraComponent>();
                    }
                }
            }
            break;
            }
            default:
                break;
            }
        });


        drawComponent<TagComponent>("Tag", entity, [this](auto& component) {
            ImGui::Text("Entity Name:");
            ImGui::SameLine();
            // Define a buffer large enough to hold the tag's content
            // Copy the current tag content into the buffer
            // Check if `m_tagBuffer` is initialized or if the entity's tag has changed
            if (m_needsTagUpdate || strncmp(m_tagBuffer, component.getTag().c_str(), sizeof(m_tagBuffer)) != 0) {
                strncpy(m_tagBuffer, component.getTag().c_str(), sizeof(m_tagBuffer));
                m_tagBuffer[sizeof(m_tagBuffer) - 1] = '\0'; // Null-terminate to avoid overflow
                m_needsTagUpdate = false; // Reset the flag after updating the buffer
            }
            // Use ImGui::InputText to allow editing
            if (ImGui::InputText("##Tag", m_tagBuffer, sizeof(m_tagBuffer))) {
                // If the input changes, update the component's tag
                component.setTag(m_tagBuffer);
            }
        });

        drawComponent<MaterialComponent>("Material", entity, [this, entity](MaterialComponent& component) {
            ImGui::Text("Material Properties");

            // Base Color Control
            ImGui::Text("Base Color");
            ImGui::ColorEdit4("##BaseColor", glm::value_ptr(component.albedo));

            ImGui::Text("Appearance Properties");
            bool update = false;
            update |= drawFloatControl("Emission", component.emission, 0.0f, 0.1f);
            update |= drawFloatControl("Diffuse", component.diffuse, 0.5f, 0.1f);
            update |= drawFloatControl("Specular", component.specular, 0.5f, 0.1f);
            update |= drawFloatControl("PhongExponents", component.phongExponent, 32.0f, 1.0f);

            /*
            // Emissive Factor Control
            ImGui::Text("Emissive Factor");
            ImGui::ColorEdit3("##EmissiveFactor", glm::value_ptr(component.emissiveFactor));
*/

            if (ImGui::Button("Reload Material Shader")) {
                component.reloadShader = true;
                m_context->activeScene()->onComponentUpdated(entity, component);
            }
            ImGui::Dummy(ImVec2(5.0f, 5.0f));
            ImGui::PushFont(m_editor->guiResources().font15);
            ImGui::Text("Texture");
            ImGui::PopFont();

            ImGui::Text("Source:");
            ImGui::Text("%s", component.albedoTexturePath.string().c_str());
            // Button to load texture
            if (ImGui::Button("Set Texture Image")) {
                std::vector<std::string> types{".png", ".jpg", ".bmp"};
                EditorUtils::openImportFileDialog("Load Texture", types, LayerUtils::TEXTURE_FILE, &m_loadFileFuture);
            }


            // Shader Controls
            ImGui::Text("Vertex Shader:");
            ImGui::Text("%s", component.vertexShaderName.string().c_str());
            if (ImGui::Button("Load Vertex Shader")) {
                std::vector<std::string> types{".vert"};
                EditorUtils::openImportFileDialog("Load Vertex Shader", types, LayerUtils::VERTEX_SHADER_FILE,
                                                  &m_loadFileFuture);
            }

            ImGui::Text("Fragment Shader:");
            ImGui::Text("%s", component.fragmentShaderName.string().c_str());
            if (ImGui::Button("Load Fragment Shader")) {
                std::vector<std::string> types{".frag"};
                EditorUtils::openImportFileDialog("Load Fragment Shader", types, LayerUtils::FRAGMENT_SHADER_FILE,
                                                  &m_loadFileFuture);
            }
            // Notify scene that material component has been updated
        });
        drawComponent<GaussianComponent2DGS>("Gaussian Model", entity, [this](GaussianComponent2DGS& component) {
            ImGui::Text("Gaussian Model Properties");

            // Display the number of Gaussians
            size_t gaussianCount = component.size();
            ImGui::Text("Number of Gaussians: %zu", gaussianCount);

            ImGui::Separator();

            // Button to add a new Gaussian
            if (ImGui::Button("Add Gaussian")) {
                // Default values for a new Gaussian
                glm::vec3 defaultMean(0.0f, 0.0f, 0.0f);
                glm::vec3 defaultNormal(0.0f, 0.0f, 1.0f); // Identity matrix
                glm::vec2 defaultScale(1.0f); // Identity matrix
                component.addGaussian(defaultMean, defaultNormal, defaultScale);
            }

            ImGui::SameLine();

            if (ImGui::Button("Load from file")) {
                std::vector<std::string> types{".ply"};
                EditorUtils::openImportFileDialog("Load 3DGS .ply file", types, LayerUtils::PLY_3DGS,
                                                  &m_loadFileFuture);
            }

            if (ImGui::Button("Remove All")) {
                component.removeAllGaussians();
            }
            ImGui::Spacing();


            // Iterate over each Gaussian and provide controls to modify them
            if (component.size() < 10) {
                for (size_t i = 0; i < component.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i)); // Ensure unique ID for ImGui widgets
                    // Collapsible header for each Gaussian
                    if (ImGui::CollapsingHeader(("Gaussian " + std::to_string(i)).c_str())) {
                        // Mean Position Controls
                        bool update = false;
                        update |= drawVec3Control("Position", component.positions[i], 0.0f);
                        update |= drawVec3Control("Normal", component.normals[i], 0.0f, 0.1f);
                        update |= drawVec2Control("Scale", component.scales[i], 0.0f, 0.1f);
                        // Amplitude Control
                        ImGui::Text("Appearance Properties");
                        update |= drawFloatControl("Opacity", component.opacities[i], 0.0f, 0.1f);
                        update |= drawFloatControl("Emission", component.emissions[i], 0.0f, 0.1f);
                        update |= drawFloatControl("Colors", component.colors[i], 1.0f, 0.1f);
                        update |= drawFloatControl("Diffuse", component.diffuse[i], 0.5f, 0.1f);
                        update |= drawFloatControl("Specular", component.specular[i], 0.5f, 0.1f);
                        update |= drawFloatControl("PhongExponents", component.phongExponents[i], 32.0f, 1.0f);


                        // Button to remove this Gaussian
                        ImGui::Spacing();
                        if (ImGui::Button("Remove Gaussian")) {
                            component.positions.erase(component.positions.begin() + i);
                            component.scales.erase(component.scales.begin() + i);
                            component.normals.erase(component.normals.begin() + i);
                            component.emissions.erase(component.emissions.begin() + i);
                            component.opacities.erase(component.opacities.begin() + i);
                            component.colors.erase(component.colors.begin() + i);
                            component.diffuse.erase(component.diffuse.begin() + i);
                            component.specular.erase(component.specular.begin() + i);
                            component.phongExponents.erase(component.phongExponents.begin() + i);
                            --i; // Adjust index after removal
                        }
                    }

                    ImGui::PopID(); // Pop ID for this Gaussian
                }
                return;
            }
            // For large numbers of Gaussians, show a single "selected" Gaussian
            // ----------------------------------------------------------------

            static int selectedGaussianIndex = 0; // or persist somewhere, e.g. as a class member

            // Ensure valid range
            if (selectedGaussianIndex < 0) selectedGaussianIndex = 0;
            if (selectedGaussianIndex >= (int)gaussianCount) {
                selectedGaussianIndex = (int)gaussianCount - 1;
            }

            // UI to pick which Gaussian to inspect
            ImGui::Text("Edit a Single Gaussian (Large Set)");
            ImGui::PushItemWidth(120.0f);
            ImGui::InputInt("Gaussian Index", &selectedGaussianIndex);
            ImGui::PopItemWidth();

            // Clamp again after user input
            if (selectedGaussianIndex < 0) selectedGaussianIndex = 0;
            if (selectedGaussianIndex >= (int)gaussianCount) {
                selectedGaussianIndex = (int)gaussianCount - 1;
            }

            // Navigation buttons to move up/down
            ImGui::SameLine();
            if (ImGui::ArrowButton("PrevGaussian", ImGuiDir_Left)) {
                selectedGaussianIndex--;
                if (selectedGaussianIndex < 0) selectedGaussianIndex = 0;
            }
            ImGui::SameLine();
            if (ImGui::ArrowButton("NextGaussian", ImGuiDir_Right)) {
                selectedGaussianIndex++;
                if (selectedGaussianIndex >= (int)gaussianCount) {
                    selectedGaussianIndex = (int)gaussianCount - 1;
                }
            }

            ImGui::Separator();

            // Now display and edit ONLY the selected Gaussian
            {
                size_t i = (size_t)selectedGaussianIndex;

                ImGui::Text("Selected Gaussian %d", selectedGaussianIndex + 1);

                bool update = false;
                update |= drawVec3Control("Position", component.positions[i], 0.0f);
                update |= drawVec3Control("Normal", component.normals[i], 0.0f, 0.1f);
                update |= drawVec2Control("Scale", component.scales[i], 0.0f, 0.1f);

                ImGui::Text("Appearance Properties");
                update |= drawFloatControl("Opacity", component.opacities[i], 0.0f, 0.1f);
                update |= drawFloatControl("Emission", component.emissions[i], 0.0f, 0.1f);
                update |= drawFloatControl("Colors", component.colors[i], 1.0f, 0.1f);
                update |= drawFloatControl("Diffuse", component.diffuse[i], 0.5f, 0.1f);
                update |= drawFloatControl("Specular", component.specular[i], 0.5f, 0.1f);
                update |= drawFloatControl("PhongExp", component.phongExponents[i], 32.0f, 1.0f);

                ImGui::Spacing();

                if (ImGui::Button("Remove This Gaussian")) {
                    component.positions.erase(component.positions.begin() + i);
                    component.normals.erase(component.normals.begin() + i);
                    component.scales.erase(component.scales.begin() + i);
                    component.emissions.erase(component.emissions.begin() + i);
                    component.colors.erase(component.colors.begin() + i);
                    component.diffuse.erase(component.diffuse.begin() + i);
                    component.specular.erase(component.specular.begin() + i);
                    component.phongExponents.erase(component.phongExponents.begin() + i);

                    // Adjust if we removed the last one
                    if (i >= component.size()) {
                        i = component.size() - 1;
                    }
                    selectedGaussianIndex = (int)i;
                }
            }
        });

        drawComponent<GaussianComponent>("Gaussian Model", entity, [this](auto& component) {
            ImGui::Text("Gaussian Model Properties");

            // Display the number of Gaussians
            ImGui::Text("Number of Gaussians: %zu", component.size());

            ImGui::Separator();

            // Button to add a new Gaussian
            component.addToRenderer = ImGui::Button("Add Gaussian");
            if (component.addToRenderer) {
                // Default values for a new Gaussian
                glm::vec3 defaultMean(0.0f, 0.0f, 0.0f);
                glm::vec3 defaultScale(0.3f); // Identity matrix
                glm::quat defaultQuat(1.0f, 0.0f, 0.0f, 0.0f); // Identity matrix
                float defaultOpacity = 1.0f;
                glm::vec3 color(1.0f, 0.0f, 0.0f);
                component.addGaussian(defaultMean, defaultScale, defaultQuat, defaultOpacity, color);
            }
            ImGui::SameLine();
            if (ImGui::Button("Load from file")) {
                std::vector<std::string> types{".ply"};
                EditorUtils::openImportFileDialog("Load 3DGS .ply file", types, LayerUtils::PLY_3DGS,
                                                  &m_loadFileFuture);
            }


            ImGui::Spacing();

            // Iterate over each Gaussian and provide controls to modify them
            if (component.size() < 10) {
                for (size_t i = 0; i < component.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i)); // Ensure unique ID for ImGui widgets
                    // Collapsible header for each Gaussian
                    if (ImGui::CollapsingHeader(("Gaussian " + std::to_string(i)).c_str())) {
                        // Mean Position Controls
                        component.addToRenderer |= drawVec3Control("Position", component.means[i], 0.0f);
                        component.addToRenderer |= drawVec3Control("Scale", component.scales[i], 0.0f, 0.1f);
                        component.addToRenderer |= drawVec3Control("Color", component.colors[i], 0.0f, 0.1f);
                        // Amplitude Control
                        ImGui::Text("Opacity");

                        component.addToRenderer |= ImGui::DragFloat("##Opacity", &component.opacities[i], 0.1f, 0.0f,
                                                                    10.0f);


                        // Button to remove this Gaussian
                        ImGui::Spacing();
                        if (ImGui::Button("Remove Gaussian")) {
                            component.means.erase(component.means.begin() + i);
                            component.scales.erase(component.scales.begin() + i);
                            component.opacities.erase(component.opacities.begin() + i);
                            component.rotations.erase(component.rotations.begin() + i);
                            --i; // Adjust index after removal
                        }
                    }

                    ImGui::PopID(); // Pop ID for this Gaussian
                }
            }
        });


        drawComponent<GroupComponent>("Group", entity, [this](auto& component) {
        });
    }


    /** Called once per frame **/
    void PropertiesLayer::onUIRender() {
        m_selectionContext = m_context->getSelectedEntity();
        ImVec2 window_pos = ImVec2(0.0f, m_editor->ui()->layoutConstants.uiYOffset); // Position (x, y)
        ImVec2 window_size = ImVec2(m_editor->ui()->width, m_editor->ui()->height); // Size (width, height)
        // Set window flags to remove decorations
        ImGuiWindowFlags window_flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        // Set next window position and size
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

        // Create the parent window
        ImGui::Begin("PropertiesLayer", NULL, window_flags);

        ImGui::Text("Entity Properties");
        std::shared_ptr<Scene> scene = m_context->activeScene();
        if (ImGui::Button("Delete This Entity")) {
            scene->destroyEntityRecursively(m_context->getSelectedEntity());
        }
        ImGui::SameLine();

        if (m_selectionContext) {
            drawComponents(m_selectionContext);
        }

        checkFileImportCompletion();
        checkFolderImportCompletion();

        ImGui::End();
    }

    /** Called once upon this object destruction **/
    void PropertiesLayer::onDetach() {
    }

    template <typename T>
    void PropertiesLayer::displayAddComponentEntry(const std::string& entryName) {
        if (!m_selectionContext.hasComponent<T>()) {
            if (ImGui::MenuItem(entryName.c_str())) {
                m_selectionContext.addComponent<T>();
                ImGui::CloseCurrentPopup();
            }
        }
    }

    void
    PropertiesLayer::handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo& loadFileInfo) {
        if (!loadFileInfo.path.empty()) {
            switch (loadFileInfo.filetype) {
            case LayerUtils::TEXTURE_FILE: {
                auto& materialComponent = m_selectionContext.getComponent<MaterialComponent>();
                materialComponent.albedoTexturePath = loadFileInfo.path;
                m_context->activeScene()->onComponentUpdated(m_selectionContext, materialComponent);
            }
            break;
            case LayerUtils::OBJ_FILE:
                // Load into the active scene
                if (m_selectionContext.hasComponent<MeshComponent>()) {
                    auto& meshComponent = m_selectionContext.getComponent<MeshComponent>();
                    meshComponent.meshParameters = std::make_shared<OBJFileMeshParameters>(loadFileInfo.path);
                }

                break;
            case LayerUtils::PLY_3DGS: {
                if (m_selectionContext.hasComponent<GaussianComponent2DGS>()) {
                    auto& comp = m_selectionContext.getComponent<GaussianComponent2DGS>();
                    comp.addGaussiansFromFile(loadFileInfo.path);
                }
            }
            break;
            case LayerUtils::PLY_MESH:
                if (m_selectionContext.hasComponent<MeshComponent>()) {
                    auto& meshComponent = m_selectionContext.getComponent<MeshComponent>();
                    meshComponent.meshParameters = std::make_shared<PLYFileMeshParameters>(loadFileInfo.path);
                }
                break;
            default:
                Log::Logger::getInstance()->warning("Not implemented yet");
                break;
            }

            // Copy the selected file path to wherever it's needed
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            opts.lastOpenedImportModelFolderPath = loadFileInfo.path;
            // Additional processing of the file can be done here
            Log::Logger::getInstance()->info("File selected: {}", loadFileInfo.path.filename().string());
        }
        else {
            Log::Logger::getInstance()->warning("No file selected.");
        }
    }

    void PropertiesLayer::checkFileImportCompletion() {
        if (m_loadFileFuture.valid() &&
            m_loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            LayerUtils::LoadFileInfo loadFileInfo = m_loadFileFuture.get(); // Get the result from the future
            handleSelectedFileOrFolder(loadFileInfo);
        }
    }

    void PropertiesLayer::checkFolderImportCompletion() {
        if (m_loadFolderFuture.valid() &&
            m_loadFolderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            LayerUtils::LoadFileInfo loadFileInfo = m_loadFolderFuture.get(); // Get the result from the future
            handleSelectedFileOrFolder(loadFileInfo);
        }
    }
}
