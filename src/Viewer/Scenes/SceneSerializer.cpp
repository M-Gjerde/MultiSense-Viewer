//
// Created by mgjer on 01/10/2024.
//

#include <yaml-cpp/yaml.h>

#include "Viewer/Scenes/SceneSerializer.h"

#include <Viewer/VkRender/Components/MaterialComponent.h>

#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Components/GaussianComponent.h"

namespace VkRender::Serialize {
    static std::string PolygonModeToString(VkPolygonMode mode) {
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

    static VkPolygonMode StringToPolygonMode(const std::string &modeStr) {
        if (modeStr == "Fill")
            return VK_POLYGON_MODE_FILL;
        if (modeStr == "Line")
            return VK_POLYGON_MODE_LINE;
        if (modeStr == "Point")
            return VK_POLYGON_MODE_POINT;

        // Default case, or handle unknown input
        return VK_POLYGON_MODE_FILL;
    }

    static std::string MeshDataTypeToString(MeshDataType meshDataType) {
        switch (meshDataType) {
            case OBJ_FILE:
                return "OBJ_FILE";
            case POINT_CLOUD:
                return "POINT_CLOUD";
            default:
                return "Unknown";
        }
    }

    static MeshDataType stringToMeshDataType(const std::string &modeStr) {
        if (modeStr == "OBJ_FILE")
            return OBJ_FILE;
        if (modeStr == "POINT_CLOUD")
            return POINT_CLOUD;
        // Default case, or handle unknown input
        return OBJ_FILE;
    }

    // Convert CameraType to string
    std::string cameraTypeToString(Camera::CameraType type) {
        switch (type) {
            case Camera::arcball:
                return "arcball";
            case Camera::flycam:
                return "flycam";
            default:
                throw std::invalid_argument("Unknown CameraType");
        }
    }

    // Convert string to CameraType
    Camera::CameraType stringToCameraType(const std::string &str) {
        if (str == "arcball") return Camera::arcball;
        if (str == "flycam") return Camera::flycam;
        throw std::invalid_argument("Unknown CameraType: " + str);
    }
}

namespace YAML {
    template<>
    struct convert<glm::vec3> {
        static Node encode(const glm::vec3 &rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node &node, glm::vec3 &rhs) {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }
            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            rhs.z = node[2].as<float>();
            return true;
        }
    };

    template<>
    struct convert<glm::quat> {
        static Node encode(const glm::quat &rhs) {
            Node node;
            node.push_back(rhs.w);
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node &node, glm::quat &rhs) {
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
    YAML::Emitter &operator<<(YAML::Emitter &out, const glm::vec3 &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter &operator<<(YAML::Emitter &out, const glm::vec4 &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter &operator<<(YAML::Emitter &out, const glm::quat &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    SceneSerializer::SceneSerializer(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
    }

    static void SerializeEntity(YAML::Emitter &out, Entity entity) {
        out << YAML::BeginMap;
        out << YAML::Key << "Entity";
        out << YAML::Value << "123123123";

        if (entity.hasComponent<TagComponent>()) {
            out << YAML::Key << "TagComponent";
            out << YAML::BeginMap;
            auto &tag = entity.getComponent<TagComponent>().Tag;
            out << YAML::Key << "Tag";
            out << YAML::Value << tag;
            out << YAML::EndMap;
        }

        if (entity.hasComponent<TransformComponent>()) {
            out << YAML::Key << "TransformComponent";
            out << YAML::BeginMap;
            auto &transform = entity.getComponent<TransformComponent>();
            out << YAML::Key << "Position";
            out << YAML::Value << transform.getPosition();
            out << YAML::Key << "Rotation";
            out << YAML::Value << transform.getRotationQuaternion();
            out << YAML::Key << "Scale";
            out << YAML::Value << transform.getScale();
            out << YAML::EndMap;
        }

        if (entity.hasComponent<MeshComponent>()) {
            out << YAML::Key << "MeshComponent";
            out << YAML::BeginMap;
            auto &mesh = entity.getComponent<MeshComponent>();
            out << YAML::Key << "ModelPath";
            out << YAML::Value << mesh.m_meshPath.string();
            out << YAML::Key << "MeshDataType";
            out << YAML::Value << Serialize::MeshDataTypeToString(mesh.m_type);
            out << YAML::Key << "PolygonMode";
            out << YAML::Value << Serialize::PolygonModeToString(mesh.polygonMode); // Serialize PolygonMode as a string
            out << YAML::EndMap;
        }
        if (entity.hasComponent<CameraComponent>()) {
            out << YAML::Key << "CameraComponent";
            out << YAML::BeginMap;
            auto &camera = entity.getComponent<CameraComponent>();
            out << YAML::Key << "render";
            out << YAML::Value << camera.render;
            auto &cameraProps = camera.camera;
            out << YAML::Key << "Type";
            out << YAML::Value << Serialize::cameraTypeToString(cameraProps.m_type);
            out << YAML::Key << "Width";
            out << YAML::Value << cameraProps.width();
            out << YAML::Key << "Height";
            out << YAML::Value << cameraProps.height();
            out << YAML::Key << "ZNear";
            out << YAML::Value << cameraProps.m_Znear;
            out << YAML::Key << "ZFar";
            out << YAML::Value << cameraProps.m_Zfar;
            out << YAML::Key << "FOV";
            out << YAML::Value << cameraProps.m_Fov;
            out << YAML::EndMap;
        }
        if (entity.hasComponent<MaterialComponent>()) {
            out << YAML::Key << "MaterialComponent";
            out << YAML::BeginMap;

            auto &material = entity.getComponent<MaterialComponent>();

            // Serialize baseColor (glm::vec4)
            out << YAML::Key << "BaseColor";
            out << YAML::Value << YAML::Flow << std::vector<float>{
                    material.baseColor.r, material.baseColor.g, material.baseColor.b, material.baseColor.a
            };

            // Serialize metallic factor (float)
            out << YAML::Key << "Metallic";
            out << YAML::Value << material.metallic;

            // Serialize roughness factor (float)
            out << YAML::Key << "Roughness";
            out << YAML::Value << material.roughness;

            // Serialize usesTexture flag (bool)
            out << YAML::Key << "UsesTexture";
            out << YAML::Value << material.usesVideoSource;

            // Serialize emissiveFactor (glm::vec4)
            out << YAML::Key << "EmissiveFactor";
            out << YAML::Value << YAML::Flow << std::vector<float>{
                    material.emissiveFactor.r, material.emissiveFactor.g, material.emissiveFactor.b,
                    material.emissiveFactor.a
            };

            // Serialize vertex shader name (std::filesystem::path)
            out << YAML::Key << "VertexShader";
            out << YAML::Value << material.vertexShaderName.string(); // Convert path to string

            // Serialize fragment shader name (std::filesystem::path)
            out << YAML::Key << "FragmentShader";
            out << YAML::Value << material.fragmentShaderName.string(); // Convert path to string

            // Serialize the flag for video source
            out << YAML::Key << "UsesVideoSource";
            out << YAML::Value << material.usesVideoSource;

            // If video source is used, serialize the video folder source
            if (material.usesVideoSource) {
                out << YAML::Key << "VideoFolderSource";
                out << YAML::Value << material.videoFolderSource.string();
                out << YAML::Key << "IsDisparity";
                out << YAML::Value << material.isDisparity;
            }

            out << YAML::EndMap;
        }

        if (entity.hasComponent<PointCloudComponent>()) {
            out << YAML::Key << "PointCloudComponent";
            out << YAML::BeginMap;
            auto &component = entity.getComponent<PointCloudComponent>();
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


        out << YAML::EndMap;
    }

    void SceneSerializer::serialize(const std::filesystem::path &filePath) {
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
        for (auto entity: m_scene->m_registry.view<entt::entity>()) {
            Entity e(entity, m_scene.get());
            if (!e)
                return;
            SerializeEntity(out, e);
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;

        std::ofstream fout(filePath);
        fout << out.c_str();
        Log::Logger::getInstance()->info("Saved scene: {} to {}", filePath.filename().string(), filePath.string());
    }

    void SceneSerializer::serializeRuntime(const std::filesystem::path &filePath) {
        throw std::runtime_error("Not implemented");
    }


    bool SceneSerializer::deserialize(const std::filesystem::path &filePath) {
        // TODO sanitize input
        std::ifstream stream(filePath);
        std::stringstream stringStream;
        stringStream << stream.rdbuf();

        YAML::Node data = YAML::Load(stringStream.str());
        if (!data["Scene"])
            return false;
        std::string sceneName = data["Scene"].as<std::string>();
        Log::Logger::getInstance()->info("Deserializing scene {}", sceneName);
        auto entities = data["Entities"];
        if (entities) {
            for (auto entity: entities) {
                uint64_t entityId = entity["Entity"].as<uint64_t>(); // todo uuid
                std::string name;
                auto tagComponent = entity["TagComponent"];
                if (tagComponent)
                    name = tagComponent["Tag"].as<std::string>();

                Entity deserializedEntity = m_scene->createEntity(name); // TOdo uuid

                auto transformComponent = entity["TransformComponent"];
                if (transformComponent) {
                    auto &tc = deserializedEntity.getComponent<TransformComponent>();
                    tc.setPosition(transformComponent["Position"].as<glm::vec3>());
                    tc.setRotationQuaternion(transformComponent["Rotation"].as<glm::quat>());
                    tc.setScale(transformComponent["Scale"].as<glm::vec3>());
                }
                auto cameraComponent = entity["CameraComponent"];
                if (cameraComponent) {
                    auto &camera = deserializedEntity.addComponent<CameraComponent>();
                    camera().setType(Serialize::stringToCameraType(cameraComponent["Type"].as<std::string>()));
                }

                auto meshComponent = entity["MeshComponent"];
                if (meshComponent) {
                    std::filesystem::path path(meshComponent["ModelPath"].as<std::string>());
                    MeshDataType meshDataType = Serialize::stringToMeshDataType(
                            meshComponent["MeshDataType"].as<std::string>());

                    auto &mesh = deserializedEntity.addComponent<MeshComponent>(path, meshDataType);
                    mesh.polygonMode = Serialize::StringToPolygonMode(
                            meshComponent["PolygonMode"].as<std::string>());

                }

                auto materialComponent = entity["MaterialComponent"];
                if (materialComponent) {
                    auto &material = deserializedEntity.addComponent<MaterialComponent>();
                    // Deserialize base color
                    auto baseColor = materialComponent["BaseColor"].as<std::vector<float> >();
                    if (baseColor.size() == 4) {
                        material.baseColor = glm::vec4(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                    }
                    // Deserialize metallic factor
                    material.metallic = materialComponent["Metallic"].as<float>();
                    // Deserialize roughness factor
                    material.roughness = materialComponent["Roughness"].as<float>();
                    // Deserialize uses texture flag
                    material.usesVideoSource = materialComponent["UsesTexture"].as<bool>();
                    // Deserialize emissive factor
                    auto emissiveFactor = materialComponent["EmissiveFactor"].as<std::vector<float> >();
                    if (emissiveFactor.size() == 4) {
                        material.emissiveFactor = glm::vec4(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2],
                                                            emissiveFactor[3]);
                    }
                    // Deserialize vertex shader name
                    if (materialComponent["VertexShader"]) {
                        material.vertexShaderName = std::filesystem::path(
                                materialComponent["VertexShader"].as<std::string>());
                    }
                    // Deserialize fragment shader name
                    if (materialComponent["FragmentShader"]) {
                        material.fragmentShaderName = std::filesystem::path(
                                materialComponent["FragmentShader"].as<std::string>());
                    }
                    // Deserialize albedo texture path (only if usesTexture is true)
                    if (material.usesVideoSource && materialComponent["VideoFolderSource"]) {
                        material.videoFolderSource = std::filesystem::path(
                                materialComponent["VideoFolderSource"].as<std::string>());
                        material.isDisparity =
                                materialComponent["IsDisparity"].as<bool>();
                    }
                }

                auto pointCloudComponent = entity["PointCloudComponent"];
                if (pointCloudComponent) {
                    auto &component = deserializedEntity.addComponent<PointCloudComponent>();
                    component.pointSize = pointCloudComponent["PointSize"].as<float>();

                    component.usesVideoSource = pointCloudComponent["UsesVideoSource"].as<bool>();
                    component.depthVideoFolderSource = std::filesystem::path(
                            pointCloudComponent["DepthVideoFolderSource"].as<std::string>());
                    component.colorVideoFolderSource = std::filesystem::path(
                            pointCloudComponent["ColorVideoFolderSource"].as<std::string>());
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

            }
        }


        Log::Logger::getInstance()->info("Loaded scene: {} from {}", filePath.filename().string(), filePath.string());

        return true;
    }

    bool SceneSerializer::deserializeRuntime(const std::filesystem::path &filePath) {
        // Not implement
        throw std::runtime_error("Not implemented");
        return false;
    }
}
