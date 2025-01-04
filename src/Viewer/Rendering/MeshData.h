//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHDATA_H
#define MULTISENSE_VIEWER_MESHDATA_H


#include "IMeshParameters.h"
#include "Components/Components.h"
#include "Viewer/Rendering/Core/RenderDefinitions.h"

namespace VkRender {
    class CylinderMeshParameters;

    enum MeshDataType : uint32_t {
        EMPTY = 0,
        OBJ_FILE = 1,
        PLY_FILE = 2,
        CYLINDER = 3,
        CAMERA_GIZMO_PERSPECTIVE = 4,
        CAMERA_GIZMO_PINHOLE = 5,
        MAX_NUM_TYPES = 6
    };

    static std::array<MeshDataType, MAX_NUM_TYPES> meshDataTypeToArray() {
        return {
            EMPTY,
            OBJ_FILE,
            PLY_FILE,
            CYLINDER,
            CAMERA_GIZMO_PERSPECTIVE,
            CAMERA_GIZMO_PINHOLE,
        };
    };

    // Const char * type for easy compatibility with imgui
    static std::array<const char*, MAX_NUM_TYPES> meshDataTypeToStringArray() {
        return {
            "EMPTY",
            "OBJ_FILE",
            "PLY_FILE",
            "CYLINDER",
            "CAMERA_GIZMO_PERSPECTIVE",
            "CAMERA_GIZMO_PINHOLE",
        };
    };

    static std::string meshDataTypeToString(MeshDataType meshDataType) {
        switch (meshDataType) {
        case EMPTY:
            return "EMPTY";
        case OBJ_FILE:
            return "OBJ_FILE";
        case PLY_FILE:
            return "PLY_FILE";
        case CAMERA_GIZMO_PERSPECTIVE:
            return "CAMERA_GIZMO_PERSPECTIVE";
        case CAMERA_GIZMO_PINHOLE:
            return "CAMERA_GIZMO_PINHOLE";
        case CYLINDER:
            return "CYLINDER";
        default:
            return "Unknown";
        }
    }

    static MeshDataType stringToMeshDataType(const std::string& modeStr) {
        if (modeStr == "EMPTY")
            return EMPTY;
        if (modeStr == "OBJ_FILE")
            return OBJ_FILE;
        if (modeStr == "CAMERA_GIZMO_PERSPECTIVE")
            return CAMERA_GIZMO_PERSPECTIVE;
        if (modeStr == "CAMERA_GIZMO_PINHOLE")
            return CAMERA_GIZMO_PINHOLE;
        if (modeStr == "PLY_FILE")
            return PLY_FILE;
        if (modeStr == "CYLINDER")
            return CYLINDER;
        // Default case, or handle unknown input
        return EMPTY;
    }


    struct MeshData {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        // Constructors
        MeshData() = default;

        // From in-memory data
        MeshData(std::vector<Vertex> vertices, std::vector<uint32_t> indices)
            : vertices(std::move(vertices)), indices(std::move(indices)) {
        }


        void generateCylinderMesh(const CylinderMeshParameters& parameters);
        void generateCameraPinholeGizmoMesh(const CameraGizmoPinholeMeshParameters& parameters);
        void generateCameraPerspectiveGizmoMesh(const CameraGizmoPerspectiveMeshParameters& parameters);
        void generateOBJMesh(const OBJFileMeshParameters& parameters);
        void generatePLYMesh(const PLYFileMeshParameters& parameters);

        bool isDirty = true;
        bool isDynamic = false;

        void computeNormals();
    };
}

#endif //MULTISENSE_VIEWER_MESHDATA_H
