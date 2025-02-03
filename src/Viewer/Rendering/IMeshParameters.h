//
// Created by magnus-desktop on 11/27/24.
//

#ifndef IMESHPARAMETERS_H
#define IMESHPARAMETERS_H

#include <memory>
#include <string>
#include <filesystem>
#include <bits/fs_path.h>
#include <glm/vec3.hpp>
#include <utility>
#include <Viewer/Application/ApplicationConfig.h>

#include "Viewer/Rendering/Core/UUID.h"
#include "Viewer/Rendering/Components/CameraComponent.h"

namespace VkRender {
    class MeshData;

    class IMeshParameters {
    public:
        virtual ~IMeshParameters() = default;
        virtual std::string getIdentifier() const = 0;
        virtual std::shared_ptr<MeshData> generateMeshData() const = 0;

    protected:
        UUID uuid;
    };


    class CylinderMeshParameters : public IMeshParameters {
    public:
        glm::vec3 origin;
        glm::vec3 direction;
        float magnitude;
        float radius = 0.05f;

        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "Cylinder_" + std::to_string(uuid);
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    class CameraGizmoPinholeMeshParameters : public IMeshParameters {
    public:
        PinholeParameters parameters;
        std::string getIdentifier() const override {
            return "CameraGizmoPinhole_" + std::to_string(uuid);
        }
        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    class CameraGizmoPerspectiveMeshParameters : public IMeshParameters {
    public:
        ProjectionParameters parameters;
        std::string getIdentifier() const override {
            return "CameraGizmoPerspective_" + std::to_string(uuid);
        }
        std::shared_ptr<MeshData> generateMeshData() const override;
    };
    class OBJFileMeshParameters : public IMeshParameters {
    public:
        explicit OBJFileMeshParameters(std::filesystem::path  path) : path(path) {
            std::filesystem::path assetsPath = ApplicationConfig::getInstance().getUserSetting().assetsPath;
            // Check if the provided path is relative to the assets path
            if (path.string().find(assetsPath) == 0) {
                // Compute the relative path from assetsPath
                relativeAssetPath = std::filesystem::relative(path, assetsPath);
            } else {
                // Log a warning or set relativeAssetPath to an empty path if it's not valid
                relativeAssetPath.clear();
            }
        }
        std::filesystem::path path;
        std::filesystem::path relativeAssetPath;

        std::string getIdentifier() const override {
            return "OBJFileMeshParameters_" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    class PLYFileMeshParameters : public IMeshParameters {
    public:
        explicit PLYFileMeshParameters(std::filesystem::path  path) : path(std::move(path)) {}

        std::filesystem::path path;

        std::string getIdentifier() const override {
            return "PLYFileMeshParameters" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    }
#endif //IMESHPARAMETERS_H
