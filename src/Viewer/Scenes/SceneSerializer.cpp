//
// Created by mgjer on 01/10/2024.
//

#include <yaml-cpp/yaml.h>

#include "Viewer/Scenes/SceneSerializer.h"

#include <Viewer/Application/ApplicationConfig.h>
#include <Viewer/Rendering/Components/MaterialComponent.h>

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/GaussianComponent.h"

namespace VkRender::Serialize {
    static std::string polygonModeToString(VkPolygonMode mode) {
        switch (mode) {
        case VK_POLYGON_MODE_FILL:
            return "Fill";
        case VK_POLYGON_MODE_LINE:
            return "Line";
        case VK_POLYGON_MODE_POINT:
            return "Point";
        default:
            return "Unknown";
        }
    }

    static VkPolygonMode stringToPolygonMode(const std::string& modeStr) {
        if (modeStr == "Fill")
            return VK_POLYGON_MODE_FILL;
        if (modeStr == "Line")
            return VK_POLYGON_MODE_LINE;
        if (modeStr == "Point")
            return VK_POLYGON_MODE_POINT;

        // Default case, or handle unknown input
        return VK_POLYGON_MODE_FILL;
    }

    /*
    // Convert CameraType to string
    std::string cameraTypeToString(Camera::CameraType type) {
        switch (type) {
            case Camera::arcball:
                return "arcball";
            case Camera::flycam:
                return "flycam";
            case Camera::pinhole:
                return "pinhole";
            default:
                throw std::invalid_argument("Unknown CameraType");
        }
    }

    // Convert string to CameraType
    Camera::CameraType stringToCameraType(const std::string &str) {
        if (str == "arcball") return Camera::arcball;
        if (str == "flycam") return Camera::flycam;
        if (str == "pinhole") return Camera::pinhole;
        throw std::invalid_argument("Unknown CameraType: " + str);
    }
    */
}

namespace YAML {
    template <>
    struct convert<glm::vec3> {
        static Node encode(const glm::vec3& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node& node, glm::vec3& rhs) {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }
            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            rhs.z = node[2].as<float>();
            return true;
        }
    };

    template <>
    struct convert<glm::quat> {
        static Node encode(const glm::quat& rhs) {
            Node node;
            node.push_back(rhs.w);
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node& node, glm::quat& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }
            rhs.w = node[0].as<float>();
            rhs.x = node[1].as<float>();
            rhs.y = node[2].as<float>();
            rhs.z = node[3].as<float>();
            return true;
        }
    };
}

namespace VkRender {
    YAML::Emitter& operator<<(YAML::Emitter& out, const glm::vec3& v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter& operator<<(YAML::Emitter& out, const glm::vec4& v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter& operator<<(YAML::Emitter& out, const glm::quat& v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    SceneSerializer::SceneSerializer(const std::shared_ptr<Scene>& scene) : m_scene(scene) {
    }

    static void SerializeEntity(YAML::Emitter& out, Entity entity) {
        out << YAML::BeginMap;
        out << YAML::Key << "Entity";
        out << YAML::Value << entity.getUUID().operator std::string();
        // Serialize Parent UUID if the entity has a ParentComponent
        if (entity.hasComponent<ParentComponent>()) {
            auto parentEntity = entity.getParent();
            out << YAML::Key << "Parent";
            out << YAML::Value << parentEntity.getUUID().operator std::string();
        }
        if (entity.hasComponent<TagComponent>()) {
            out << YAML::Key << "TagComponent";
            out << YAML::BeginMap;
            auto& tag = entity.getComponent<TagComponent>().Tag;
            out << YAML::Key << "Tag";
            out << YAML::Value << tag;
            out << YAML::EndMap;
        }
        if (entity.hasComponent<TransformComponent>()) {
            out << YAML::Key << "TransformComponent";
            out << YAML::BeginMap;
            auto& transform = entity.getComponent<TransformComponent>();
            out << YAML::Key << "Position";
            out << YAML::Value << transform.getPosition();
            out << YAML::Key << "Rotation";
            out << YAML::Value << transform.getRotationQuaternion();
            out << YAML::Key << "Scale";
            out << YAML::Value << transform.getScale();
            out << YAML::EndMap;
        }
        // Serialize VisibleComponent
        if (entity.hasComponent<VisibleComponent>()) {
            out << YAML::Key << "VisibleComponent";
            out << YAML::BeginMap;
            auto& visible = entity.getComponent<VisibleComponent>().visible;
            out << YAML::Key << "Visible";
            out << YAML::Value << visible;
            out << YAML::EndMap;
        }
        // Serialize GroupComponent
        if (entity.hasComponent<GroupComponent>()) {
            out << YAML::Key << "GroupComponent";
            out << YAML::BeginMap;
            // Add any group-specific serialization if needed
            out << YAML::EndMap;
        }
        if (entity.hasComponent<MeshComponent>()) {
            out << YAML::Key << "MeshComponent";
            out << YAML::BeginMap;
            auto& mesh = entity.getComponent<MeshComponent>();
            switch (mesh.meshDataType()) {
            case OBJ_FILE:
                {
                    auto params = std::dynamic_pointer_cast<OBJFileMeshParameters>(mesh.meshParameters);
                    out << YAML::Key << "ModelPath";
                    out << YAML::Value << params->path.string();
                }
                break;
            case PLY_FILE:
                {
                    auto params = std::dynamic_pointer_cast<PLYFileMeshParameters>(mesh.meshParameters);
                    out << YAML::Key << "ModelPath";
                    out << YAML::Value << params->path.string();
                }
                break;
            default:
                break;
            }
            out << YAML::Key << "MeshDataType";
            out << YAML::Value << meshDataTypeToString(mesh.meshDataType());
            out << YAML::Key << "PolygonMode";
            out << YAML::Value << Serialize::polygonModeToString(mesh.polygonMode());
            // Serialize PolygonMode as a string
            out << YAML::EndMap;
        }

        if (entity.hasComponent<CameraComponent>()) {
            out << YAML::Key << "CameraComponent";
            out << YAML::BeginMap;
            auto &camera = entity.getComponent<CameraComponent>();
            auto type = camera.cameraType;
            // Serialize CameraType
            out << YAML::Key << "CameraType";
            out << YAML::Value << CameraComponent::cameraTypeToString(camera.cameraType);

            // Serialize based on CameraType
            switch (camera.cameraType) {
                case CameraComponent::ARCBALL:
                    // ARCBALL-specific serialization (if any) can be added here
                    break;
                case CameraComponent::PERSPECTIVE: {
                    auto &params = camera.projectionParameters;
                    out << YAML::Key << "ProjectionParameters";
                    out << YAML::BeginMap;
                    out << YAML::Key << "Near" << YAML::Value << params.near;
                    out << YAML::Key << "Far" << YAML::Value << params.far;
                    out << YAML::Key << "Aspect" << YAML::Value << params.aspect;
                    out << YAML::Key << "FOV" << YAML::Value << params.fov;
                    out << YAML::EndMap;
                    break;
                }
                case CameraComponent::PINHOLE: {
                    auto &params = camera.pinHoleParameters;
                    out << YAML::Key << "PinHoleParameters";
                    out << YAML::BeginMap;
                    out << YAML::Key << "Height" << YAML::Value << params.height;
                    out << YAML::Key << "Width" << YAML::Value << params.width;
                    out << YAML::Key << "Fx" << YAML::Value << params.fx;
                    out << YAML::Key << "Fy" << YAML::Value << params.fy;
                    out << YAML::Key << "Cx" << YAML::Value << params.cx;
                    out << YAML::Key << "Cy" << YAML::Value << params.cy;
                    out << YAML::EndMap;
                    break;
                }
                default:
                    Log::Logger::getInstance()->warning("Fallback: Cannot serialize camera type");
            }
            out << YAML::EndMap;

        }

        if (entity.hasComponent<MaterialComponent>()) {
            out << YAML::Key << "MaterialComponent";
            out << YAML::BeginMap;
            auto& material = entity.getComponent<MaterialComponent>();
            // Serialize baseColor (glm::vec4)
            out << YAML::Key << "BaseColor";
            out << YAML::Value << YAML::Flow << std::vector<float>{
                material.albedo.r, material.albedo.g, material.albedo.b, material.albedo.a
            };
            // Serialize metallic factor (float)
            out << YAML::Key << "Emission";
            out << YAML::Value << material.emission;
            out << YAML::Key << "Diffuse";
            out << YAML::Value << material.diffuse;
            out << YAML::Key << "Specular";
            out << YAML::Value << material.specular;
            out << YAML::Key << "PhongExponent";
            out << YAML::Value << material.phongExponent;
            // Serialize vertex shader name (std::filesystem::path)
            out << YAML::Key << "VertexShader";
            out << YAML::Value << material.vertexShaderName.string(); // Convert path to string
            // Serialize fragment shader name (std::filesystem::path)
            out << YAML::Key << "FragmentShader";
            out << YAML::Value << material.fragmentShaderName.string(); // Convert path to string
            out << YAML::Key << "AlbedoTexturePath";
            out << YAML::Value << material.albedoTexturePath.string(); // Convert path to string
            out << YAML::EndMap;
        }

        if (entity.hasComponent<PointCloudComponent>()) {
            out << YAML::Key << "PointCloudComponent";
            out << YAML::BeginMap;
            auto& component = entity.getComponent<PointCloudComponent>();
            out << YAML::Key << "PointSize";
            out << YAML::Value << component.pointSize;
            // Serialize the flag for video source
            out << YAML::Key << "UsesVideoSource";
            out << YAML::Value << component.usesVideoSource;

            // If video source is used, serialize the video folder source
            if (component.usesVideoSource) {
                out << YAML::Key << "DepthVideoFolderSource";
                out << YAML::Value << component.depthVideoFolderSource.string();
                out << YAML::Key << "ColorVideoFolderSource";
                out << YAML::Value << component.colorVideoFolderSource.string();
            }


            out << YAML::EndMap;
        }
        if (entity.hasComponent<GaussianComponent2DGS>()) {
            out << YAML::Key << "GaussianComponent2DGS";
            auto& component = entity.getComponent<GaussianComponent2DGS>();
            out << YAML::BeginMap;
            // Serialize positions
            out << YAML::Key << "Positions";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& position : component.positions) {
                out << YAML::Flow << YAML::BeginSeq << position.x << position.y << position.z << YAML::EndSeq;
            }
            out << YAML::EndSeq;

            // Serialize normals
            out << YAML::Key << "Normals";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& normal : component.normals) {
                out << YAML::Flow << YAML::BeginSeq << normal.x << normal.y << normal.z << YAML::EndSeq;
            }
            out << YAML::EndSeq;

            // Serialize scales
            out << YAML::Key << "Scales";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& scale : component.scales) {
                out << YAML::Flow << YAML::BeginSeq << scale.x << scale.y << YAML::EndSeq;
            }
            out << YAML::EndSeq;

            // Serialize float properties
            auto serializeFloatArray = [&](const std::vector<float>& values, const std::string& key) {
                out << YAML::Key << key;
                out << YAML::Value << YAML::BeginSeq;
                for (const auto& value : values) {
                    out << value;
                }
                out << YAML::EndSeq;
            };

            serializeFloatArray(component.emissions, "Emissions");
            serializeFloatArray(component.colors, "Colors");
            serializeFloatArray(component.diffuse, "Diffuse");
            serializeFloatArray(component.specular, "Specular");
            serializeFloatArray(component.phongExponents, "PhongExponents");

            out << YAML::EndMap;
        }

        if (entity.hasComponent<GaussianComponent>()) {
            out << YAML::Key << "GaussianComponent";
            out << YAML::BeginMap;
            auto& component = entity.getComponent<GaussianComponent>();

            // Serialize the means
            out << YAML::Key << "Means";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& mean : component.means) {
                out << YAML::Flow << YAML::BeginSeq << mean.x << mean.y << mean.z << YAML::EndSeq;
            }
            out << YAML::EndSeq;

            out << YAML::Key << "Scales";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& scale : component.scales) {
                out << YAML::Flow << scale;
            }
            out << YAML::EndSeq;

            out << YAML::Key << "Rotations";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& rotation : component.rotations) {
                out << YAML::Flow << rotation;
            }
            out << YAML::EndSeq;

            // Serialize the amplitudes
            out << YAML::Key << "Opacities";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& amplitude : component.opacities) {
                out << amplitude;
            }
            out << YAML::EndSeq;

            out << YAML::Key << "Colors";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto& color : component.colors) {
                out << YAML::Flow << color;
            }
            out << YAML::EndSeq;

            out << YAML::EndMap;
        }
        if (entity.hasComponent<GroupComponent>()) {
            out << YAML::Key << "GroupComponent";
            out << YAML::BeginMap;
            auto& groupComponent = entity.getComponent<GroupComponent>();
            out << YAML::Key << "ColmapPath";
            out << YAML::Value << groupComponent.colmapPath.string();
            out << YAML::EndMap;
        }

        out << YAML::EndMap;
    }

    void SceneSerializer::serialize(const std::filesystem::path& filePath) {
        // Ensure the directory exists
        if (filePath.has_parent_path()) {
            std::filesystem::create_directories(filePath.parent_path());
        }

        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "Scene";
        out << YAML::Value << "Scene name";
        out << YAML::Key << "Entities";
        out << YAML::Value << YAML::BeginSeq;
        for (auto entity : m_scene->m_registry.view<entt::entity>()) {
            Entity e(entity, m_scene.get());
            if (!e || e.hasComponent<TemporaryComponent>())
                continue;
            SerializeEntity(out, e);
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;

        std::ofstream fout(filePath);
        fout << out.c_str();
        Log::Logger::getInstance()->info("Saved scene: {} to {}", filePath.filename().string(), filePath.string());
    }

    void SceneSerializer::serializeRuntime(const std::filesystem::path& filePath) {
        throw std::runtime_error("Not implemented");
    }


    bool SceneSerializer::deserialize(const std::filesystem::path& filePath) {
        // TODO sanitize input
        std::ifstream stream(filePath);
        std::stringstream stringStream;
        stringStream << stream.rdbuf();

        YAML::Node data = YAML::Load(stringStream.str());
        if (!data["Scene"])
            return false;
        std::string sceneName = data["Scene"].as<std::string>();
        Log::Logger::getInstance()->info("Deserializing scene: {} from: {}", sceneName, filePath.string());
        auto entities = data["Entities"];
        if (entities) {
            std::unordered_map<uint64_t, Entity> entityMap;

            for (auto entity : entities) {
                auto entityId = UUID(entity["Entity"].as<uint64_t>()); // todo uuid
                std::string name = "Unnamed";
                auto tagComponent = entity["TagComponent"];
                if (tagComponent)
                    name = tagComponent["Tag"].as<std::string>();

                Entity deserializedEntity = m_scene->createEntityWithUUID(entityId, name); // TOdo uuid

                auto transformComponent = entity["TransformComponent"];
                if (transformComponent) {
                    auto& tc = deserializedEntity.getComponent<TransformComponent>();
                    tc.setPosition(transformComponent["Position"].as<glm::vec3>());
                    tc.setRotationQuaternion(transformComponent["Rotation"].as<glm::quat>());
                    tc.setScale(transformComponent["Scale"].as<glm::vec3>());
                }

                // Deserialize VisibleComponent
                auto visibleComponentNode = entity["VisibleComponent"];
                if (visibleComponentNode) {
                    auto& visibleComponent = deserializedEntity.addComponent<VisibleComponent>();
                    visibleComponent.visible = visibleComponentNode["Visible"].as<bool>();
                }

                // Store the entity in the map
                entityMap[entityId] = deserializedEntity;


                auto cameraComponent = entity["CameraComponent"];
                if (cameraComponent) {
                    auto& camera = deserializedEntity.addComponent<CameraComponent>();

                    // Deserialize CameraType
                    if (cameraComponent["CameraType"]) {
                        std::string cameraTypeStr = cameraComponent["CameraType"].as<std::string>();
                        camera.cameraType = CameraComponent::stringToCameraType(cameraTypeStr);
                    }

                    // Deserialize based on CameraType
                    switch (camera.cameraType) {
                        case CameraComponent::ARCBALL:
                            // ARCBALL-specific deserialization (if any) can be added here
                            break;

                        case CameraComponent::PERSPECTIVE:
                        {
                            auto projectionParams = cameraComponent["ProjectionParameters"];
                            if (projectionParams) {
                                camera.projectionParameters.near = projectionParams["Near"].as<float>(0.1f);
                                camera.projectionParameters.far = projectionParams["Far"].as<float>(100.0f);
                                camera.projectionParameters.aspect = projectionParams["Aspect"].as<float>(1.6f);
                                camera.projectionParameters.fov = projectionParams["FOV"].as<float>(60.0f);
                                camera.updateParametersChanged();
                            }
                            break;
                        }

                        case CameraComponent::PINHOLE:
                        {
                            auto pinholeParams = cameraComponent["PinHoleParameters"];
                            if (pinholeParams) {
                                camera.pinHoleParameters.height = pinholeParams["Height"].as<int>(720);
                                camera.pinHoleParameters.width = pinholeParams["Width"].as<int>(1280);
                                camera.pinHoleParameters.fx = pinholeParams["Fx"].as<float>(1280.0f);
                                camera.pinHoleParameters.fy = pinholeParams["Fy"].as<float>(720.0f);
                                camera.pinHoleParameters.cx = pinholeParams["Cx"].as<float>(640.0f);
                                camera.pinHoleParameters.cy = pinholeParams["Cy"].as<float>(360.0f);
                            }
                            break;
                        }

                        default:
                            Log::Logger::getInstance()->warning("Fallback: Cannot deserialize camera type");
                    }
                    camera.updateParametersChanged();
                }

                auto meshComponent = entity["MeshComponent"];
                if (meshComponent) {
                    std::filesystem::path path;
                    if (meshComponent["ModelPath"]) {
                        path = meshComponent["ModelPath"].as<std::string>();
                    }

                    std::string meshDataTypeStr = meshComponent["MeshDataType"].as<std::string>();
                    MeshDataType meshDataType = stringToMeshDataType(meshDataTypeStr);

                    // Add MeshComponent to the entity
                    MeshComponent& mesh = deserializedEntity.addComponent<MeshComponent>(meshDataType, path);
                    // Deserialize PolygonMode
                    if (meshComponent["PolygonMode"] && meshComponent["PolygonMode"].IsScalar()) {
                        std::string polygonModeStr = meshComponent["PolygonMode"].as<std::string>();
                        mesh.polygonMode() = Serialize::stringToPolygonMode(polygonModeStr);
                    }
                    else {
                        // Handle missing PolygonMode (optional: set default or throw error)
                        mesh.polygonMode() = VK_POLYGON_MODE_FILL; // Default value
                    }
                }

                auto materialComponent = entity["MaterialComponent"];
                if (materialComponent) {
                    auto& material = deserializedEntity.addComponent<MaterialComponent>();
                    // Deserialize base color
                    auto baseColor = materialComponent["BaseColor"].as<std::vector<float>>();
                    if (baseColor.size() == 4) {
                        material.albedo = glm::vec4(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                    }
                    if (materialComponent["Emission"]) {
                        material.emission = materialComponent["Emission"].as<float>();
                    } else {
                        material.emission = 0.0f; // Default value or handle as needed
                    }
                    if (materialComponent["Diffuse"]) {
                        material.diffuse = materialComponent["Diffuse"].as<float>();
                    } else {
                        material.diffuse = 0.0f; // Default value or handle as needed
                    }
                    if (materialComponent["Specular"]) {
                        material.specular = materialComponent["Specular"].as<float>();
                    } else {
                        material.specular = 0.0f; // Default value or handle as needed
                    }
                    if (materialComponent["PhongExponent"]) {
                        material.phongExponent = materialComponent["PhongExponent"].as<float>();
                    } else {
                        material.phongExponent = 32.0f; // Default value or handle as needed
                    }
                    // Deserialize uses texture flag
                    if (materialComponent["VertexShader"]) {
                        material.vertexShaderName = std::filesystem::path(
                            materialComponent["VertexShader"].as<std::string>());
                    }
                    // Deserialize fragment shader name
                    if (materialComponent["FragmentShader"]) {
                        material.fragmentShaderName = std::filesystem::path(
                            materialComponent["FragmentShader"].as<std::string>());
                    }
                }

                auto pointCloudComponent = entity["PointCloudComponent"];
                if (pointCloudComponent) {
                    auto& component = deserializedEntity.addComponent<PointCloudComponent>();
                    component.pointSize = pointCloudComponent["PointSize"].as<float>();

                    component.usesVideoSource = pointCloudComponent["UsesVideoSource"].as<bool>();
                    component.depthVideoFolderSource = std::filesystem::path(
                        pointCloudComponent["DepthVideoFolderSource"].as<std::string>());
                    component.colorVideoFolderSource = std::filesystem::path(
                        pointCloudComponent["ColorVideoFolderSource"].as<std::string>());
                }
                auto groupComponent = entity["GroupComponent"];
                if (groupComponent) {
                    auto& component = deserializedEntity.addComponent<GroupComponent>();
                    auto colmapPathNode = groupComponent["ColmapPath"];
                    if (colmapPathNode)
                        component.colmapPath = colmapPathNode.as<std::string>();
                }

                auto gaussianComponentNode = entity["GaussianComponent"];
                if (gaussianComponentNode) {
                    auto& component = deserializedEntity.addComponent<GaussianComponent>();

                    // Deserialize means
                    auto meansNode = gaussianComponentNode["Means"];
                    if (meansNode) {
                        for (const auto& meanNode : meansNode) {
                            glm::vec3 mean;
                            mean.x = meanNode[0].as<float>();
                            mean.y = meanNode[1].as<float>();
                            mean.z = meanNode[2].as<float>();
                            component.means.push_back(mean);
                        }
                    }

                    auto covariancesNode = gaussianComponentNode["Scales"];
                    if (covariancesNode) {
                        for (const auto& covNode : covariancesNode) {
                            component.scales.push_back(covNode.as<glm::vec3>());
                        }
                    }
                    auto rotationsNode = gaussianComponentNode["Rotations"];
                    if (rotationsNode) {
                        for (const auto& rotNode : rotationsNode) {
                            component.rotations.push_back(rotNode.as<glm::quat>());
                        }
                    }

                    // Deserialize amplitudes
                    auto amplitudesNode = gaussianComponentNode["Opacities"];
                    if (amplitudesNode) {
                        for (const auto& amplitudeNode : amplitudesNode) {
                            float amplitude = amplitudeNode.as<float>();
                            component.opacities.push_back(amplitude);
                        }
                    }
                    // Deserialize amplitudes
                    auto colorsNode = gaussianComponentNode["Colors"];
                    if (colorsNode) {
                        for (const auto& colorNode : colorsNode) {
                            auto color = colorNode.as<glm::vec3>();
                            component.colors.push_back(color);
                        }
                    }
                }

                auto gaussianComponent2DGSNode = entity["GaussianComponent2DGS"];
                if (gaussianComponent2DGSNode) {
                    auto& component = deserializedEntity.addComponent<GaussianComponent2DGS>();
                    auto& node = gaussianComponent2DGSNode;
                    // Deserialize positions
                    if (node["Positions"]) {
                        for (const auto& positionNode : node["Positions"]) {
                            glm::vec3 position(
                                    positionNode[0].as<float>(),
                                    positionNode[1].as<float>(),
                                    positionNode[2].as<float>()
                            );
                            component.positions.push_back(position);
                        }
                    }

                    // Deserialize normals
                    if (node["Normals"]) {
                        for (const auto& normalNode : node["Normals"]) {
                            glm::vec3 normal(
                                    normalNode[0].as<float>(),
                                    normalNode[1].as<float>(),
                                    normalNode[2].as<float>()
                            );
                            component.normals.push_back(normal);
                        }
                    }

                    // Deserialize scales
                    if (node["Scales"]) {
                        for (const auto& scaleNode : node["Scales"]) {
                            glm::vec2 scale(
                                    scaleNode[0].as<float>(),
                                    scaleNode[1].as<float>()
                            );
                            component.scales.push_back(scale);
                        }
                    }
                    // Deserialize float properties
                    auto deserializeFloatArray = [&](std::vector<float>& values, const std::string& key) {
                        if (node[key]) {
                            for (const auto& valueNode : node[key]) {
                                values.push_back(valueNode.as<float>());
                            }
                        }
                    };
                    deserializeFloatArray(component.emissions, "Emissions");
                    deserializeFloatArray(component.colors, "Colors");
                    deserializeFloatArray(component.diffuse, "Diffuse");
                    deserializeFloatArray(component.specular, "Specular");
                    deserializeFloatArray(component.phongExponents, "PhongExponents");
                }
            }

            for (auto entityNode : entities) {
                uint64_t uuid = entityNode["Entity"].as<uint64_t>();
                Entity deserializedEntity = entityMap[uuid];

                // Check if the entity has a parent
                auto parentUUIDNode = entityNode["Parent"];
                if (parentUUIDNode) {
                    uint64_t parentUUID = parentUUIDNode.as<uint64_t>();
                    if (entityMap.find(parentUUID) != entityMap.end()) {
                        Entity parentEntity = entityMap[parentUUID];
                        deserializedEntity.setParent(parentEntity);
                    }
                    else {
                        Log::Logger::getInstance()->warning("Parent entity with UUID {} not found.", parentUUID);
                    }
                }
            }
        }


        Log::Logger::getInstance()->info("Loaded scene: {} from {}", filePath.filename().string(), filePath.string());

        return true;
    }

    bool SceneSerializer::deserializeRuntime(const std::filesystem::path& filePath) {
        // Not implement
        throw std::runtime_error("Not implemented");
        return false;
    }
}
