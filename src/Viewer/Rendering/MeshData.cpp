//
// Created by magnus on 11/27/24.
//


#include "Viewer/Rendering/MeshData.h"

#include "tinyply.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#define TINYOBJLOADER_USE_MAPBOX_EARCUT

#include <tiny_obj_loader.h>
#include <stb_image.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/hash.hpp>
#include "glm/ext.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <utility>

#include "IMeshParameters.h"


namespace VkRender {

    void MeshData::generateCylinderMesh(const CylinderMeshParameters& parameters) {
        // Extract parameters from the map
        glm::vec3 origin = parameters.origin;
        glm::vec3 direction =  parameters.direction;
        float magnitude =  parameters.magnitude;

        // Parameters for the cylinder
        const int segments = 20; // Adjust for smoother cylinder
        const float radius = parameters.radius; // Adjust as needed
        const float height = magnitude;

        glm::vec3 endPoint = origin + direction * height;

        // Generate the cylinder vertices and indices
        // Define the base circle and top circle vertices
        std::vector<Vertex> baseCircleVertices;
        std::vector<Vertex> topCircleVertices;

        for (int i = 0; i < segments; ++i) {
            float theta = 2.0f * glm::pi<float>() * float(i) / float(segments);
            float x = radius * cos(theta);
            float z = radius * sin(theta);

            glm::vec3 offset = glm::vec3(x, 0.0f, z);

            // Rotate offset to align with the direction vector
            glm::vec3 defaultUp = glm::vec3(0.0f, 1.0f, 0.0f);
            glm::quat rotationQuat = glm::rotation(defaultUp, glm::normalize(direction));

            // Rotate the offset
            offset = rotationQuat * offset;

            // Base vertex
            Vertex baseVertex{};
            baseVertex.pos = origin + offset;
            baseVertex.normal = -direction;
            baseVertex.uv0 = glm::vec2(float(i) / segments, 0.0f);
            baseCircleVertices.push_back(baseVertex);

            // Top vertex
            Vertex topVertex{};
            topVertex.pos = endPoint + offset;
            topVertex.normal = direction;
            topVertex.uv0 = glm::vec2(float(i) / segments, 1.0f);
            topCircleVertices.push_back(topVertex);
        }

        // Combine vertices
        vertices.reserve(segments * 2);
        vertices.insert(vertices.end(), baseCircleVertices.begin(), baseCircleVertices.end());
        vertices.insert(vertices.end(), topCircleVertices.begin(), topCircleVertices.end());

        // Generate indices for the side faces
        for (int i = 0; i < segments; ++i) {
            int next = (i + 1) % segments;
            int baseIndex = i;
            int topIndex = i + segments;
            int nextBaseIndex = next;
            int nextTopIndex = next + segments;

            // First triangle of quad
            indices.push_back(baseIndex);
            indices.push_back(nextBaseIndex);
            indices.push_back(topIndex);

            // Second triangle of quad
            indices.push_back(nextBaseIndex);
            indices.push_back(nextTopIndex);
            indices.push_back(topIndex);
        }

        // Generate indices for the base and top caps if desired
        // Base cap

        for (int i = 1; i < segments - 1; ++i) {
            indices.push_back(0);
            indices.push_back(i);
            indices.push_back(i + 1);
        }

        // Top cap
        for (int i = 1; i < segments - 1; ++i) {
            indices.push_back(segments);
            indices.push_back(segments + i + 1);
            indices.push_back(segments + i);
        }


        isDynamic = true;
    }


    void MeshData::generateCameraPinholeGizmoMesh(const CameraGizmoPinholeMeshParameters& pinhole) {
        float width  = pinhole.parameters.width;
        float height = pinhole.parameters.height;
        float fx     = pinhole.parameters.fx;
        float fy     = pinhole.parameters.fy;
        float cx     = pinhole.parameters.cx;
        float cy     = pinhole.parameters.cy;
        float focalLength     = pinhole.parameters.focalLength;

        // Choose a plane at Z = -1 for visualization. Objects in front of the camera have negative Z.

        // Convert image corners into camera coordinates:
        // We'll map four corners of the image:
        // Top-left:     (u,v) = (0, 0)
        // Top-right:    (u,v) = (width, 0)
        // Bottom-right: (u,v) = (width, height)
        // Bottom-left:  (u,v) = (0, height)
        //
        // The mapping chosen so that:
        // X = -(u - cx)*Z_plane/fx
        // Y =  (v - cy)*Z_plane/fy
        // Z =  Z_plane (negative)
        //
        // This ensures:
        // - Left side (u<cx) => negative X
        // - Top side (v<cy) => positive Y
        // - Forward along negative Z

        auto mapPixelTo3D = [&](float u, float v, float Z_plane) {
            float X = -(u - cx) * Z_plane / fx;
            float Y = -(v - cy) * Z_plane / fy; // Notice the minus sign before (v - cy)
            float Z = Z_plane;
            return glm::vec3(X, Y, Z);
        };

        float Z_plane = -1.0f;

        glm::vec3 frontTopLeft     = mapPixelTo3D(0.0f,     0.0f, Z_plane);
        glm::vec3 frontTopRight    = mapPixelTo3D(width,     0.0f, Z_plane);
        glm::vec3 frontBottomRight = mapPixelTo3D(width,     height, Z_plane);
        glm::vec3 frontBottomLeft  = mapPixelTo3D(0.0f,      height, Z_plane);

        glm::vec3 backTopLeft     = mapPixelTo3D(0.0f,     0.0f, focalLength / 1000); // convert focal length to meters
        glm::vec3 backTopRight    = mapPixelTo3D(width,     0.0f, focalLength / 1000); // convert focal length to meters
        glm::vec3 backBottomRight = mapPixelTo3D(width,     height, focalLength / 1000); // convert focal length to meters
        glm::vec3 backBottomLeft  = mapPixelTo3D(0.0f,      height, focalLength / 1000); // convert focal length to meters

        // The pinhole (camera center) at the origin
        glm::vec3 pinholePos = glm::vec3(0.0f, 0.0f, 0.0f);

        // Define vertices: pinhole + the four corners of the image plane
        // We'll store them in order: pinhole(0), topLeft(1), topRight(2), bottomRight(3), bottomLeft(4)
        std::vector<glm::vec3> uboVertices = {
                pinholePos,
                frontTopLeft,
                frontTopRight,
                frontBottomRight,
                frontBottomLeft,
                backTopLeft,
                backTopRight,
                backBottomRight,
                backBottomLeft
        };

        // We'll create a simple line-based mesh:
        // Lines from pinhole to each corner:
        // pinhole -> topLeft
        // pinhole -> topRight
        // pinhole -> bottomRight
        // pinhole -> bottomLeft
        //
        // And lines forming the image plane rectangle:
        // topLeft -> topRight
        // topRight -> bottomRight
        // bottomRight -> bottomLeft
        // bottomLeft -> topLeft

        indices = {
                // Lines from pinhole (0) to image corners
                0, 1,
                0, 2,
                0, 3,
                0, 4,

                // Image plane rectangle
                1, 2,
                2, 3,
                3, 4,
                4, 1,

                // Lines from pinhole (0) to sensor corners
                0, 5,
                0, 6,
                0, 7,
                0, 8,

                // Image plane rectangle
                5, 8, 7,
                5, 7, 6,

        };

        vertices.resize(uboVertices.size());
        for (size_t i = 0; i < uboVertices.size(); ++i) {
            vertices[i].pos    = uboVertices[i];
            vertices[i].normal = glm::vec3(0.0f, 0.0f, 1.0f); // Normal not very meaningful for a line gizmo
            vertices[i].uv0    = glm::vec2(0.0f, 0.0f);       // Placeholder UV
            vertices[i].uv1    = glm::vec2(0.0f, 0.0f);       // Placeholder UV
            vertices[i].color  = glm::vec4(1.0f);             // White color
        }

        // This is a gizmo; often drawn as lines. Ensure rendering mode is line-friendly if needed.
        isDynamic = true;
    }

    void MeshData::generateCameraPerspectiveGizmoMesh(const CameraGizmoPerspectiveMeshParameters& perspective) {
        float nearDist = perspective.parameters.near;
        float farDist = perspective.parameters.far;
        float fovDegrees = perspective.parameters.fov;
        float aspect = perspective.parameters.aspect;

        // Convert FOV to radians if it's given in degrees
        float fovRadians = glm::radians(fovDegrees);
        float tanHalfFov = std::tan(fovRadians * 0.5f);

        // Compute the half-widths and half-heights of the near and far planes
        float nearHeight = 2.0f * nearDist * tanHalfFov;
        float nearWidth  = nearHeight * aspect;

        float farHeight = 2.0f * farDist * tanHalfFov;
        float farWidth  = farHeight * aspect;

        // Define the vertices for a frustum:
        // Near plane corners (Z = nearDist)
        // We'll assume the camera looks along +Z, with +X to the right and +Y up.
        glm::vec3 NBL(-nearWidth * 0.5f, -nearHeight * 0.5f, -nearDist); // Near Bottom Left
        glm::vec3 NBR( nearWidth * 0.5f, -nearHeight * 0.5f, -nearDist); // Near Bottom Right
        glm::vec3 NTR( nearWidth * 0.5f,  nearHeight * 0.5f, -nearDist); // Near Top Right
        glm::vec3 NTL(-nearWidth * 0.5f,  nearHeight * 0.5f, -nearDist); // Near Top Left

        // Far plane corners (Z = farDist)
        glm::vec3 FBL(-farWidth * 0.5f, -farHeight * 0.5f, -farDist); // Far Bottom Left
        glm::vec3 FBR( farWidth * 0.5f, -farHeight * 0.5f, -farDist); // Far Bottom Right
        glm::vec3 FTR( farWidth * 0.5f,  farHeight * 0.5f, -farDist); // Far Top Right
        glm::vec3 FTL(-farWidth * 0.5f,  farHeight * 0.5f, -farDist); // Far Top Left

        std::vector<glm::vec3> uboVertices = {
                NBL, NBR, NTR, NTL, // 0-3 : Near plane
                FBL, FBR, FTR, FTL  // 4-7 : Far plane
        };

        // We will define the frustum as a closed mesh with 6 faces, each face made of two triangles.
        // Indices order (triangles) for each quad is typically: (0,1,2) and (2,3,0).
        indices = {
                // Near face
                0, 1, 2,
                2, 3, 0,

                // Far face
                4, 5, 6,
                6, 7, 4,

                // Left face (NBL, NTL, FTL, FBL)
                0, 3, 7,
                7, 4, 0,

                // Right face (NBR, NTR, FTR, FBR)
                1, 2, 6,
                6, 5, 1,

                // Top face (NTL, NTR, FTR, FTL)
                3, 2, 6,
                6, 7, 3,

                // Bottom face (NBL, NBR, FBR, FBL)
                0, 1, 5,
                5, 4, 0
        };

        // Resize vertex array to match our vertex count
        vertices.resize(uboVertices.size());

        for (size_t i = 0; i < uboVertices.size(); ++i) {
            vertices[i].pos     = uboVertices[i];
            vertices[i].normal  = glm::vec3(0.0f, 0.0f, 1.0f); // Placeholder normal
            vertices[i].uv0     = glm::vec2(0.0f, 0.0f);       // Placeholder UV
            vertices[i].uv1     = glm::vec2(0.0f, 0.0f);       // Placeholder UV
            vertices[i].color   = glm::vec4(1.0f);             // White color
        }

        isDynamic = true;
    }

    void MeshData::generateOBJMesh(const OBJFileMeshParameters& parameters) {
                tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;


        if (!reader.ParseFromFile(parameters.path.string(), reader_config)) {
            if (!reader.Error().empty()) {
                //std::cerr << "TinyObjReader: " << reader.Error();
                Log::Logger::getInstance()->error("Failed to load .OBJ file {}", parameters.path.string());
            }
            return;
        }


        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ file empty {}", parameters.path.string());
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        // Pre-allocate memory for vertices and indices vectors
        size_t estimatedVertexCount = attrib.vertices.size() / 3;
        size_t estimatedIndexCount = 0;
        for (const auto& shape : shapes) {
            estimatedIndexCount += shape.mesh.indices.size();
        }

        std::unordered_map<VkRender::Vertex, uint32_t> uniqueVertices{};
        uniqueVertices.reserve(estimatedVertexCount); // Reserve space for unique vertices

        vertices.reserve(estimatedVertexCount); // Reserve space to avoid resizing
        indices.reserve(estimatedIndexCount); // Reserve space to avoid resizing

        std::vector<glm::vec3> tempNormals(vertices.size(), glm::vec3(0.0f));

        bool calculateNormals = false;

        // Build the mesh (vertices + indices)
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                VkRender::Vertex vertex{};

                // Position
                vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                };

                // Normal (if available). Otherwise, set to zero to mark that we need to compute it
                if (index.normal_index > -1) {
                    vertex.normal = {
                            attrib.normals[3 * index.normal_index + 0],
                            attrib.normals[3 * index.normal_index + 1],
                            attrib.normals[3 * index.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 0.0f, 0.0f}; // Will compute later
                }

                // UV
                if (index.texcoord_index > -1) {
                    vertex.uv0 = {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                }

                // De-duplicate
                auto [it, inserted] = uniqueVertices.try_emplace(vertex, static_cast<uint32_t>(vertices.size()));
                if (inserted) {
                    vertices.push_back(vertex);
                }

                indices.push_back(it->second);
            }
        }

        // Now compute normals for any vertices that have a zero normal
        computeNormals();
    }

    void MeshData::computeNormals()
    {
        // Accumulator array the same size as `vertices`
        std::vector<glm::vec3> accumulators(vertices.size(), glm::vec3(0.0f));

        // Accumulate face normals
        for (size_t i = 0; i < indices.size(); i += 3) {
            // Indices of the triangle
            uint32_t i0 = indices[i + 0];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            // Positions of the triangle’s vertices
            const glm::vec3& p0 = vertices[i0].pos;
            const glm::vec3& p1 = vertices[i1].pos;
            const glm::vec3& p2 = vertices[i2].pos;

            // Compute face normal (using right-handed cross product)
            glm::vec3 edge1 = p1 - p0;
            glm::vec3 edge2 = p2 - p0;
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

            // Accumulate the face normal in each of the triangle’s vertices
            accumulators[i0] += faceNormal;
            accumulators[i1] += faceNormal;
            accumulators[i2] += faceNormal;
        }

        // Assign / normalize
        for (size_t i = 0; i < vertices.size(); i++) {
            // If the vertex had a normal of (0,0,0) or we choose to override it,
            // we use the accumulated normal. Otherwise, you can skip to retain existing normals.
            if (glm::length(vertices[i].normal) < 1e-6f) {
                vertices[i].normal = glm::normalize(accumulators[i]);
            }
        }
    }

    void MeshData::generatePLYMesh(const PLYFileMeshParameters& parameters) {
                std::ifstream ss(parameters.path.string(), std::ios::binary);
        if (ss.fail()) {
            Log::Logger::getInstance()->warning("Failed to open {}", parameters.path.string());
            return;
        }

        tinyply::PlyFile file;
        file.parse_header(ss);

        std::shared_ptr<tinyply::PlyData> positionData, colorData, facesData;

        try {
            // Request position data (double type)
            positionData = file.request_properties_from_element("vertex", {"x", "y", "z"});
        }
        catch (const std::exception& e) {
            Log::Logger::getInstance()->warning("Failed to Vertex information from {}", parameters.path.string());
        }

        try {
            // Request color data (uchar type)
            colorData = file.request_properties_from_element("vertex", {"red", "green", "blue"});
        }
        catch (const std::exception& e) {
            Log::Logger::getInstance()->warning("Failed to Vertex color information from {}", parameters.path.string());
        }

        try {
            facesData = file.request_properties_from_element("face", {"vertex_indices"});

            facesData = file.request_properties_from_element(
                "face",
                { "vertex_indices" },
                3
                );

        }
        catch (const std::exception& e) {
            Log::Logger::getInstance()->warning("Failed to face/vertex_indices information from {}", parameters.path.string());
        }

        file.read(ss);

        const size_t numVertices = positionData->count;
        const size_t numFaces = facesData->count;

        // Load position data (double precision)
        std::vector<double> positions(numVertices * 3);
        std::memcpy(positions.data(), positionData->buffer.get(), positionData->buffer.size_bytes());

        // Load color data (unsigned char precision)
        std::vector<uint8_t> colors(numVertices * 3, 1);
        if (colorData) {
            std::memcpy(colors.data(), colorData->buffer.get(), colorData->buffer.size_bytes());
        }
        std::vector<uint32_t> faces(numFaces * 3);
        if (facesData) {
            std::memcpy(faces.data(), facesData->buffer.get(), facesData->buffer.size_bytes());
        }

        // Populate vertices
        for (size_t i = 0; i < numVertices; ++i) {
            Vertex vertex{};
            vertex.pos = {
                static_cast<float>(positions[3 * i + 0]),
                static_cast<float>(positions[3 * i + 1]),
                static_cast<float>(positions[3 * i + 2])
            };

            // Convert color values from uint8 to float [0, 1]
            vertex.color = {
                colors[3 * i + 0] / 255.0f,
                colors[3 * i + 1] / 255.0f,
                colors[3 * i + 2] / 255.0f,
                1.0f // Alpha channel set to 1.0
            };

            vertex.normal = glm::vec3(0.0f);
            vertices.push_back(vertex);
        }

        // Populate indices
        for (size_t i = 0; i < numFaces; ++i) {
            indices.push_back(faces[3 * i + 0]);
            indices.push_back(faces[3 * i + 1]);
            indices.push_back(faces[3 * i + 2]);
        }

        // Compute face normals and accumulate them in each vertex normal
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i + 0];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            glm::vec3 v0 = vertices[i0].pos;
            glm::vec3 v1 = vertices[i1].pos;
            glm::vec3 v2 = vertices[i2].pos;

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

            vertices[i0].normal += faceNormal;
            vertices[i1].normal += faceNormal;
            vertices[i2].normal += faceNormal;
        }

        // Normalize all the vertex normals
        for (auto& vertex : vertices) {
            vertex.normal = glm::normalize(vertex.normal);
        }

    }
}
