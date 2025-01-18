//
// Created by magnus on 1/18/25.
//

#include "Viewer/Rendering/Components/GaussianComponent.h"
#include <cstring>


namespace VkRender{


    void GaussianComponent2DGS::loadFromPly(const std::filesystem::path &path) {
        try {
            // Open the .ply file in binary mode
            std::ifstream fileStream(path, std::ios::binary);
            if (!fileStream.is_open()) {
                throw std::runtime_error("Unable to open file: " + path.string());
            }

            // Parse header
            tinyply::PlyFile plyFile;
            plyFile.parse_header(fileStream);

            // Request the properties we care about.
            // Make sure these match your Python exporter’s header exactly!
            std::shared_ptr<tinyply::PlyData> vertexData =
                    plyFile.request_properties_from_element("vertex", {"x", "y", "z"});

            std::shared_ptr<tinyply::PlyData> rotationData =
                    plyFile.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

            std::shared_ptr<tinyply::PlyData> scaleData =
                    plyFile.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});

            std::shared_ptr<tinyply::PlyData> colorData =
                    plyFile.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});

            std::shared_ptr<tinyply::PlyData> opacityData =
                    plyFile.request_properties_from_element("vertex", {"opacity"});

            // Read all requested data
            plyFile.read(fileStream);

            // Ensure all buffers have the same vertex count
            const size_t count = vertexData->count;
            if (rotationData->count != count || scaleData->count != count || opacityData->count != count ||
                colorData->count != count) {
                throw std::runtime_error("Inconsistent vertex count among requested properties.");
            }

            // Copy raw floats out of tinyply’s memory buffers
            std::vector<float> verticesPLY(count * 3);
            std::memcpy(verticesPLY.data(), vertexData->buffer.get(), count * 3 * sizeof(float));

            std::vector<float> rotationsPLY(count * 4);
            std::memcpy(rotationsPLY.data(), rotationData->buffer.get(), count * 4 * sizeof(float));

            std::vector<float> scalesPLY(count * 3);
            std::memcpy(scalesPLY.data(), scaleData->buffer.get(), count * 3 * sizeof(float));

            std::vector<float> opacitiesPLY(count);
            std::memcpy(opacitiesPLY.data(), opacityData->buffer.get(), count * sizeof(float));

            std::vector<float> colorsPLY(count * 3);
            std::memcpy(colorsPLY.data(), colorData->buffer.get(), count * 3 * sizeof(float));

            // Reserve space in our arrays
            reserve(count);

            // Convert to glm-friendly data
            for (size_t i = 0; i < count; ++i) {
                glm::vec3 pos(
                        verticesPLY[i * 3 + 0],
                        verticesPLY[i * 3 + 1],
                        verticesPLY[i * 3 + 2]
                );

                glm::vec3 color(
                        colorsPLY[i * 3 + 0],
                        colorsPLY[i * 3 + 1],
                        colorsPLY[i * 3 + 2]
                );

                glm::quat quat(
                        // Quat constructor is (w, x, y, z) by default in GLM,
                        // but your data might be (rot_0, rot_1, rot_2, rot_3) = (x, y, z, w)?
                        // Adjust as needed:
                        rotationsPLY[i * 4 + 0],  // w
                        rotationsPLY[i * 4 + 1],  // x
                        rotationsPLY[i * 4 + 2],  // y
                        rotationsPLY[i * 4 + 3]   // z
                );
                //glm::vec3 normal = quat * glm::vec3(0, 0, 1) * glm::conjugate(quat);
                glm::vec3 normal = quat * glm::vec3(0, 0, 1);
                normal = -normal; // Flip the normal direction

                // If your data is stored (w, x, y, z) in the .ply, then
                //   glm::quat(rotations[i*4+0], rotations[i*4+1], rotations[i*4+2], rotations[i*4+3]);
                // or reorder to match exactly.

                glm::vec2 scale(
                        glm::exp(scalesPLY[i * 3 + 0]),
                        glm::exp(scalesPLY[i * 3 + 1])
                        //,scalesPLY[i * 3 + 2]
                );

                float opacityVal = opacitiesPLY[i];


                addGaussian(pos, normal, scale, opacityVal);
            }

            std::cout << "Successfully loaded " << count << " entries from " << path << std::endl;
        }
        catch (const std::exception &e) {
            std::cerr << "Error loading PLY file: " << e.what() << std::endl;
        }
    }
}