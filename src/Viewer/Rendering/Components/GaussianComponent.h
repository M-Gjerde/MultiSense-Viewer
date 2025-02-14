//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
#define MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H

#include <tinyply.h>
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

        std::vector<float> opacities;        // Contiguous array for amplitudes
        std::vector<float> emissions;        // Contiguous array for amplitudes
        std::vector<glm::vec4> colors;        // Contiguous array for amplitudes
        std::vector<float> diffuse;        // Contiguous array for amplitudes
        std::vector<float> specular;        // Contiguous array for amplitudes
        std::vector<float> phongExponents;        // Contiguous array for amplitudes

        // Resize to hold n Gaussians
        void resize(size_t n) {
            positions.resize(n);
            scales.resize(n);
            normals.resize(n);
            opacities.resize(n);
            emissions.resize(n);
            colors.resize(n);
            diffuse.resize(n);
            specular.resize(n);
            phongExponents.resize(n);
        }
        // reserve to hold n Gaussians
        void reserve(size_t n) {
            positions.reserve(n);
            scales.reserve(n);
            normals.reserve(n);
            opacities.reserve(n);
            emissions.reserve(n);
            colors.reserve(n);
            diffuse.reserve(n);
            specular.reserve(n);
            phongExponents.reserve(n);
        }

        void removeAllGaussians(){
            positions.clear();
            scales.clear();
            normals.clear();
            opacities.clear();
            emissions.clear();
            colors.clear();
            diffuse.clear();
            specular.clear();
            phongExponents.clear();
        }
        // Add a Gaussian with default values for float properties and emission set to 0
        void addGaussian(const glm::vec3 &position,
                         const glm::vec3 &normal,
                         const glm::vec2 &scale,
                         float opacity = 1.0f,
                         glm::vec4 color = glm::vec4(glm::vec3(0.0f), 1.0f),
                         float diffuseValue = 0.5f,
                         float specularValue = 0.5f,
                         float phongExponent = 32.0f) {
            positions.push_back(position);
            normals.push_back(normal);
            scales.push_back(scale);

            opacities.push_back(opacity);           // Set emission to 0
            emissions.push_back(0.0f);           // Set emission to 0
            colors.push_back(color);
            diffuse.push_back(diffuseValue);
            specular.push_back(specularValue);
            phongExponents.push_back(phongExponent);
        }

        void addGaussiansFromFile(const std::filesystem::path &plyFilePath) {
            loadFromPly(plyFilePath);
        }


        // Get the number of Gaussians
        size_t size() const {
            return positions.size();
        }

    private:
        void loadFromPly(const std::filesystem::path &path);
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
                         glm::vec3 color, const std::array<std::array<float, 15>, 3> &sh = {}) {
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


    };
}
#endif //MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
