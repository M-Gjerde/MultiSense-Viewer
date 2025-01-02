//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
#define MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H

#include <multisense_viewer/external/tinyply/source/tinyply.h>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace VkRender {

    struct GaussianComponent2DGS {
        std::vector<glm::vec3> positions;         // Contiguous array for mean positions
        std::vector<glm::vec3> normals;   // Contiguous array for covariance matrices
        std::vector<glm::vec2> scales;   // Contiguous array for covariance matrices

        std::vector<float> emissions;        // Contiguous array for amplitudes
        std::vector<float> colors;        // Contiguous array for amplitudes
        std::vector<float> diffuse;        // Contiguous array for amplitudes
        std::vector<float> specular;        // Contiguous array for amplitudes
        std::vector<float> phongExponents;        // Contiguous array for amplitudes

        // Resize to hold n Gaussians
        void resize(size_t n) {
            positions.resize(n);
            scales.resize(n);
            normals.resize(n);
            emissions.resize(n);
            colors.resize(n);
            diffuse.resize(n);
            specular.resize(n);
            phongExponents.resize(n);
        }

        // Add a Gaussian with default values for float properties and emission set to 0
        void addGaussian(const glm::vec3 &position,
                         const glm::vec3 &normal,
                         const glm::vec2 &scale,
                         float color = 1.0f,
                         float diffuseValue = 0.5f,
                         float specularValue = 0.5f,
                         float phongExponent = 32.0f)
        {
            positions.push_back(position);
            normals.push_back(normal);
            scales.push_back(scale);

            emissions.push_back(0.0f);           // Set emission to 0
            colors.push_back(color);
            diffuse.push_back(diffuseValue);
            specular.push_back(specularValue);
            phongExponents.push_back(phongExponent);
        }


        // Get the number of Gaussians
        size_t size() const {
            return positions.size();
        }
    };

    struct GaussianComponent {
        std::vector<glm::vec3> means;         // Contiguous array for mean positions
        std::vector<glm::vec3> scales;   // Contiguous array for covariance matrices
        std::vector<float> opacities;        // Contiguous array for amplitudes
        std::vector<glm::quat> rotations;
        std::vector<glm::vec3> colors;   // Contiguous array for covariance matrices
        std::vector<std::array<std::array<float, 15>, 3>> shCoeffs;
        bool addToRenderer = true;

        GaussianComponent() = default;

        explicit GaussianComponent(std::filesystem::path pathToPly) {
            loadFromPly(pathToPly);
        }

        // Resize to hold n Gaussians
        void resize(size_t n) {
            means.resize(n);
            scales.resize(n);
            opacities.resize(n);
            rotations.resize(n);
            colors.resize(n);
            shCoeffs.resize(n);
        }

        // Add a Gaussian
        void addGaussian(const glm::vec3 &mean, const glm::vec3 &scale, const glm::quat &rotation, float opacity,
                         glm::vec3 color, const std::array<std::array<float, 15>, 3>& sh = {}) {
            means.push_back(mean);
            scales.push_back(scale);
            opacities.push_back(opacity);
            rotations.push_back(rotation);
            colors.push_back(color);
            shCoeffs.push_back(sh);

        }

        // Get the number of Gaussians
        size_t size() const {
            return means.size();
        }


    private:
        void loadFromPly(const std::filesystem::path &path) {
            try {
                std::ifstream fileStream(path.string(), std::ios::binary);
                if (!fileStream) {
                    throw std::runtime_error("Unable to open file.");
                }

                // Read the header to extract property names
                std::vector<std::string> propertyNames;
                std::vector<std::string> extraFNames;
                std::string line;
                std::streampos dataStartPos;
                while (std::getline(fileStream, line)) {
                    if (line == "end_header") {
                        dataStartPos = fileStream.tellg();
                        break;
                    }

                    std::istringstream iss(line);
                    std::string token;
                    iss >> token;
                    if (token == "property") {
                        std::string type, name;
                        iss >> type >> name;
                        propertyNames.push_back(name);

                        if (name.find("f_rest_") == 0) {
                            extraFNames.push_back(name);
                        }
                    }
                }

                // Sort extraFNames by the integer suffix
                std::sort(extraFNames.begin(), extraFNames.end(),
                          [](const std::string &a, const std::string &b) {
                              int idx_a = std::stoi(a.substr(7)); // "f_rest_".length() == 7
                              int idx_b = std::stoi(b.substr(7));
                              return idx_a < idx_b;
                          });

                // Reset the fileStream to the beginning
                fileStream.clear();
                fileStream.seekg(0, std::ios::beg);

                // Now, proceed with tinyply
                tinyply::PlyFile file;
                file.parse_header(fileStream);

                // Define buffers
                std::vector<float> vertices, scales, opacities, rotations, colors, featuresExtra;

                std::shared_ptr<tinyply::PlyData> vertexData, scaleData, opacityData, rotationData, colorData, extraFeatureData;

                // Request properties
                vertexData = file.request_properties_from_element("vertex", {"x", "y", "z"});
                colorData = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
                opacityData = file.request_properties_from_element("vertex", {"opacity"});
                scaleData = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
                rotationData = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

                // Request extra features if they exist
                if (!extraFNames.empty()) {
                    extraFeatureData = file.request_properties_from_element("vertex", extraFNames);
                }

                // Read all the data
                file.read(fileStream);

                // Extract vertices
                vertices.assign((float *)vertexData->buffer.get(),
                                (float *)vertexData->buffer.get() + vertexData->count * 3);
                scales.assign((float *)scaleData->buffer.get(),
                              (float *)scaleData->buffer.get() + scaleData->count * 3);
                rotations.assign((float *)rotationData->buffer.get(),
                                 (float *)rotationData->buffer.get() + rotationData->count * 4);
                opacities.assign((float *)opacityData->buffer.get(),
                                 (float *)opacityData->buffer.get() + opacityData->count);
                colors.assign((float *)colorData->buffer.get(),
                              (float *)colorData->buffer.get() + colorData->count * 3);

                // Process the scales with exponential
                for (size_t i = 0; i < scaleData->count; ++i) {
                    scales[i * 3 + 0] = std::exp(scales[i * 3 + 0]);
                    scales[i * 3 + 1] = std::exp(scales[i * 3 + 1]);
                    scales[i * 3 + 2] = std::exp(scales[i * 3 + 2]);
                }

                // Apply sigmoid to opacities
                for (size_t i = 0; i < opacityData->count; ++i) {
                    opacities[i] = 1.0f / (1.0f + std::exp(-opacities[i]));
                }

                // Process extra features if available
                std::vector<std::vector<std::vector<float>>> featuresExtraReshaped;
                if (extraFeatureData) {
                    size_t vertexCount = extraFeatureData->count;
                    size_t totalCoeffs = extraFNames.size();

                    featuresExtra.assign((float *)extraFeatureData->buffer.get(),
                                         (float *)extraFeatureData->buffer.get() + vertexCount * totalCoeffs);

                    // Reshape featuresExtra to [vertexCount, 3, num_coeffs_per_channel]
                    int num_coeffs_per_channel = totalCoeffs / 3;
                    featuresExtraReshaped.resize(vertexCount, std::vector<std::vector<float>>(3, std::vector<float>(num_coeffs_per_channel)));

                    for (size_t i = 0; i < vertexCount; ++i) {
                        for (int c = 0; c < 3; ++c) {
                            for (int k = 0; k < num_coeffs_per_channel; ++k) {
                                featuresExtraReshaped[i][c][k] = featuresExtra[i * totalCoeffs + c * num_coeffs_per_channel + k];
                            }
                        }
                    }
                }

                // Add each Gaussian from the read data
                for (size_t i = 0; i < vertexData->count; ++i) {
                    glm::vec3 mean(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
                    glm::vec3 scale(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]);
                    glm::quat rotation(rotations[i * 4 + 0], rotations[i * 4 + 1], rotations[i * 4 + 2], rotations[i * 4 + 3]);
                    rotation = glm::normalize(rotation);

                    float opacity = opacities[i];
                    glm::vec3 color_dc(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);

                    std::array<std::array<float, 15>, 3> sh; // 3 channels

                    // Include SH coefficients if available
                    if (extraFeatureData) {
                        int num_coeffs_per_channel = extraFNames.size() / 3;
                        if (num_coeffs_per_channel != 15){
                            throw std::runtime_error("Ply file does not contain 15 coefficients per channel");
                        }
                        for (int c = 0; c < 3; ++c) {
                            for (int k = 0; k < num_coeffs_per_channel; ++k) {
                                sh[c][k] = featuresExtraReshaped[i][c][k];
                            }
                        }
                    }
                    addGaussian(mean, scale, rotation, opacity, color_dc, sh);
                }
            } catch (const std::exception &e) {
                std::cerr << "Error loading PLY file: " << e.what() << std::endl;
            }
        }
    };
}
#endif //MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
