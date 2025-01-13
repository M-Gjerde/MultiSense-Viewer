//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACERKERNELS_H
#define MULTISENSE_VIEWER_RAYTRACERKERNELS_H

#include <sycl/sycl.hpp>
#include <Viewer/Rendering/Components/CameraComponent.h>
#include <Viewer/Rendering/Components/TransformComponent.h>

#include "Definitions.h"


namespace VkRender::RT::Kernels {
    static bool rayTriangleIntersect(
            glm::vec3 &ray_origin,
            glm::vec3 &ray_dir,
            const glm::vec3 &a,
            const glm::vec3 &b,
            const glm::vec3 &c,
            glm::vec3 &out_intersection) {
        float epsilon = 1e-7f;

        glm::vec3 edge1 = b - a;
        glm::vec3 edge2 = c - a;
        glm::vec3 h = glm::cross(ray_dir, edge2);
        float det = glm::dot(edge1, h);

        // If det is near zero, ray is parallel to triangle
        if (det > -epsilon && det < epsilon)
            return false;

        float inv_det = 1.0f / det;
        glm::vec3 s = ray_origin - a;
        float u = inv_det * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return false;

        glm::vec3 q = glm::cross(s, edge1);
        float v = inv_det * glm::dot(ray_dir, q);
        if (v < 0.0f || (u + v) > 1.0f)
            return false;

        float t = inv_det * glm::dot(edge2, q);
        if (t > epsilon) {
            out_intersection = ray_origin + ray_dir * t;
            return true;
        }

        return false;
    }



    class RenderKernel {
    public:
        RenderKernel(GPUData gpuData, uint32_t width, uint32_t height, uint32_t size, TransformComponent cameraPose,
                     PinholeCamera* camera)
                : m_gpuData(gpuData), m_width(width), m_height(height), m_size(size), m_cameraTransform(cameraPose),
                  m_camera(camera) {
        }

        void operator()(sycl::nd_item<2> item) const {
            uint32_t x = item.get_global_id(1);
            uint32_t y = item.get_global_id(0);

            uint32_t pixelIndex = (y * m_width + x) * 4;
            if (pixelIndex >= m_size)
                return;

            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float Z_plane = -1.0f;

            auto mapPixelTo3D = [&](float u, float v) {
                float X = -(u - cx) * Z_plane / fx;
                float Y = -(v - cy) * Z_plane / fy; // Notice the minus sign before (v - cy)
                float Z = Z_plane;
                return glm::vec3(X, Y, Z);
            };
            glm::vec3 direction = mapPixelTo3D(static_cast<float>(x), static_cast<float>(y));
            glm::vec3 rayOrigin = m_cameraTransform.translation;
            glm::vec3 worldRayDir = glm::normalize(glm::mat3(m_cameraTransform.rotation) * glm::normalize(direction));
            // Primary ray intersection
            float closest_t = FLT_MAX;
            bool hit = false;
            size_t hitIdx = 0; // Index of the Gaussian that is hit
            glm::vec3 hitPointWorld(0.0f);

            // Loop over all entities
            for (uint32_t entityIdx = 0; entityIdx < m_gpuData.numEntities; ++entityIdx) {
                // Get entity transform and invert it to transform the ray into local space
                glm::mat4 entityTransform  = m_gpuData.transforms[entityIdx].getTransform();
                glm::mat4 invEntityTransform = glm::inverse(entityTransform);

                // Transform ray to local space
                glm::vec3 localRayOrigin = glm::vec3(invEntityTransform * glm::vec4(rayOrigin, 1.0f));
                glm::vec3 localRayDir    = glm::normalize(
                        glm::vec3(invEntityTransform * glm::vec4(worldRayDir, 0.0f))
                );

                // Determine which slice of the index buffer belongs to this entity
                uint32_t startIndex = m_gpuData.indexOffsets[entityIdx];
                // If this is the last entity, go until the end. Otherwise, go until next offset.
                uint32_t endIndex   = (entityIdx + 1 < m_gpuData.numEntities)
                                      ? m_gpuData.indexOffsets[entityIdx + 1]
                                      : 0; // We'll fix this below

                // If the user never stored a total index count, you can do:
                //  - either store totalIndices in GPUData
                //  - or handle it carefully.
                // For clarity, let's assume 'endIndex' is:
                if (entityIdx + 1 < m_gpuData.numEntities) {
                    endIndex = m_gpuData.indexOffsets[entityIdx + 1];
                } else {
                    // If we have a total index count, use it
                    endIndex = m_gpuData.totalIndices; // if you had stored it
                    // If not, you need to pass it from the host or keep track in some other way.
                    // For example, assume indexOffsets.back() is the last offset:
                    // endIndex = m_gpuData.indexOffsets[m_gpuData.numEntities - 1] + ???
                    // We'll illustrate one approach:
                    // endIndex = <the final number of indices uploaded>;
                }

                // For simplicity, let's assume you have a "totalIndices" or you set endIndex properly:
                // size_t entityIndexCount = endIndex - startIndex;
                // size_t triangleCount    = entityIndexCount / 3;

                // If you do NOT have totalIndices stored, you might do:
                // (We'll assume 'endIndex' is computed or we do a minimal example.)
                // For demonstration, let's do a safer pattern check:
                if (endIndex <= startIndex) {
                    // This means the current entity is the last one or misconfigured
                    // Without totalIndices in GPUData, you might not know how far to go
                    continue;
                }
                size_t entityIndexCount = endIndex - startIndex;
                size_t triangleCount    = entityIndexCount / 3;

                // Test each triangle
                for (size_t t = 0; t < triangleCount; ++t) {
                    uint32_t i0 = m_gpuData.indices[startIndex + t * 3 + 0];
                    uint32_t i1 = m_gpuData.indices[startIndex + t * 3 + 1];
                    uint32_t i2 = m_gpuData.indices[startIndex + t * 3 + 2];

                    // Local-space vertices
                    const glm::vec3& aLocal = m_gpuData.vertices[i0].position;
                    const glm::vec3& bLocal = m_gpuData.vertices[i1].position;
                    const glm::vec3& cLocal = m_gpuData.vertices[i2].position;

                    // Ray-triangle test in local space
                    glm::vec3 localHit(0.0f);
                    if (rayTriangleIntersect(localRayOrigin, localRayDir, aLocal, bLocal, cLocal, localHit)) {
                        // Convert the local hit point back to world space
                        glm::vec3 worldHit = glm::vec3(entityTransform * glm::vec4(localHit, 1.0f));
                        float dist = glm::distance(rayOrigin, worldHit);

                        if (dist < closest_t) {
                            closest_t     = dist;
                            hit           = true;
                            hitPointWorld = worldHit;
                        }
                    }
                }
            } // end for (entityIdx)


            if (hit) {
                // If we hit a triangle, color the pixel accordingly.
                // For a simple visualization, let’s map intersection distance to grayscale.
                float maxDistance = 10.0f;
                float minDistance = 0.25f;

                // Clamp the distance to the range [minDistance, maxDistance]
                float clampedDist = glm::clamp(closest_t, minDistance, maxDistance);

                // Map the clamped distance to the intensity range [0, 255]
                uint8_t intensity = static_cast<uint8_t>((clampedDist - minDistance) / (maxDistance - minDistance) * 255.0f);

                // Write the intensity to the image memory (RGBA)
                m_gpuData.imageMemory[pixelIndex + 0] = 255 - intensity; // R
                m_gpuData.imageMemory[pixelIndex + 1] = 255 - intensity; // G
                m_gpuData.imageMemory[pixelIndex + 2] = 255 - intensity; // B
                m_gpuData.imageMemory[pixelIndex + 3] = 255;       // A
            }
            else {
                // No intersection: clear pixel to some background color.
                m_gpuData.imageMemory[pixelIndex + 0] = 0; // R
                m_gpuData.imageMemory[pixelIndex + 1] = 0; // G
                m_gpuData.imageMemory[pixelIndex + 2] = 0; // B
                m_gpuData.imageMemory[pixelIndex + 3] = 255; // A
            }

            /*
                static bool
    intersectPlane(const glm::vec3 &n, const glm::vec3 &p0, const glm::vec3 &l0, const glm::vec3 &l, float &t) {
        // Assuming vectors are normalized
        float denom = glm::dot(n, l);
        if (std::abs(denom) > 1e-6) { // Avoid near-parallel cases
            glm::vec3 p0l0 = p0 - l0;
            t = glm::dot(p0l0, n) / denom;
            return (t >= 0); // Only return true for intersections in front of the ray
        }
        return false;
    }
            for (size_t idx = 0; idx < m_gpuData.numGaussians; ++idx) {
                glm::vec3 &pos = m_gpuData.gaussianInputAssembly[idx].position;
                glm::vec3 &normal = m_gpuData.gaussianInputAssembly[idx].normal;
                glm::vec2 &scale = m_gpuData.gaussianInputAssembly[idx].scale;
                float intensity = m_gpuData.gaussianInputAssembly[idx].intensity;

                float t = FLT_MAX;
                if (intersectPlane(normal, pos, rayOrigin, worldRayDir, t) && t < closest_t) {
                    closest_t = t;
                    hit = true;
                    hitIdx = idx;

                }
            }


            float pixelIntensity = 0.0f;

            if (hit) {
                // Retrieve hit Gaussian parameters
                glm::vec3 &pos = m_gpuData.gaussianInputAssembly[hitIdx].position;
                glm::vec3 &normal = m_gpuData.gaussianInputAssembly[hitIdx].normal;
                glm::vec2 &scale = m_gpuData.gaussianInputAssembly[hitIdx].scale;
                float intensity = m_gpuData.gaussianInputAssembly[hitIdx].intensity;

                // Compute intersection point
                glm::vec3 intersectionPoint = rayOrigin + closest_t * worldRayDir;

                glm::vec3 u = pos - intersectionPoint;
                glm::vec3 v = glm::cross(u, normal);

                float alpha = glm::length(u);
                float beta = glm::length(v);

                if ((alpha < scale.x) * 3 && (beta * 3) < scale.y) {
                    // Evaluate the Gaussian:
                    // G(alpha, beta) = intensity * exp(-0.5 * ((alpha^2 / scale.x^2) + (beta^2 / scale.y^2)))
                    float gaussVal = intensity * std::exp(-0.5f * ((alpha * alpha) / (scale.x * scale.x)
                                                                   + (beta * beta) / (scale.y * scale.y)));

                    pixelIntensity = gaussVal * 255;

                }
            }


            if (hit) {
                m_gpuData.imageMemory[pixelIndex + 0] = pixelIntensity; // R
                m_gpuData.imageMemory[pixelIndex + 1] = pixelIntensity; // G
                m_gpuData.imageMemory[pixelIndex + 2] = pixelIntensity; // B
                m_gpuData.imageMemory[pixelIndex + 3] = 255; // A}
            } else {
                // No intersection: clear pixel to some background color.
                m_gpuData.imageMemory[pixelIndex + 0] = 0; // R
                m_gpuData.imageMemory[pixelIndex + 1] = 0; // G
                m_gpuData.imageMemory[pixelIndex + 2] = 0; // B
                m_gpuData.imageMemory[pixelIndex + 3] = 255; // A
            }
             */
        }

    private:
        GPUData m_gpuData;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_size;

        TransformComponent m_cameraTransform;
        PinholeCamera* m_camera;
    };
}


#endif //MULTISENSE_VIEWER_RAYTRACERKERNELS_H
