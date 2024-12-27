//
// Created by magnus on 12/27/24.
//

#ifndef LIGHTTRACERKERNELS_H
#define LIGHTTRACERKERNELS_H

#include "Viewer/Rendering/RenderResources/Raytracer/Definitions.h"

namespace VkRender::RT {
class RenderKernelLightTracing {
public:
    RenderKernelLightTracing(GPUData        gpuData,
                             uint32_t       numPhotons,
                             uint32_t       width,
                             uint32_t       height,
                             uint32_t       size,
                             TransformComponent cameraPose,
                             PinholeCamera  camera,
                             uint32_t       maxBounces)
        : m_gpuData(gpuData)
        , m_numPhotons(numPhotons)
        , m_width(width)
        , m_height(height)
        , m_size(size)
        , m_cameraTransform(cameraPose)
        , m_camera(camera)
        , m_maxBounces(maxBounces)
    {}

    void operator()(sycl::item<1> item) const {
        size_t photonID = item.get_linear_id();
        if (photonID >= m_numPhotons) {
            return;
        }

        // Each thread traces one photon.
        traceOnePhoton(photonID);
    }

private:
    GPUData         m_gpuData{};
    uint32_t        m_numPhotons{};
    uint32_t        m_width;
    uint32_t        m_height;
    uint32_t        m_size;
    uint32_t        m_maxBounces;  // e.g. 5, 8, or 10

    TransformComponent m_cameraTransform;
    PinholeCamera      m_camera;

    // ---------------------------------------------------------
    // Single Photon Trace (Multi-Bounce)
    // ---------------------------------------------------------
    void traceOnePhoton(size_t photonID) const
    {
        // -----------------------------------------------------
        // 1) Sample Emissive Patch & Direction
        // -----------------------------------------------------
        size_t lightIdx = sampleRandomEmissivePatch(photonID);
        const GaussianInputAssembly& lightPatch = m_gpuData.gaussianInputAssembly[lightIdx];

        glm::vec3 rayOrigin = lightPatch.position;
        glm::vec3 rayDir    = sampleEmissionDirection(photonID, lightPatch.normal);

        // The photon *throughput* tracks how much flux remains after each bounce
        float throughput = lightPatch.intensity;

        // -----------------------------------------------------
        // 2) Loop over Bounces (including direct sensor check)
        // -----------------------------------------------------
        for (uint32_t bounce = 0; bounce < m_maxBounces; ++bounce) {
            // -------------------------------------------------
            // A) Check if photon hits sensor (before geometry)
            // -------------------------------------------------
            float tSensor = FLT_MAX;
            bool  sensorHit = intersectPlane(
                                  m_gpuData.sensorPlaneNormal,
                                  m_gpuData.sensorPlanePos,
                                  rayOrigin,
                                  rayDir,
                                  tSensor);

            // We'll track the closest intersection in the scene
            float tMin   = FLT_MAX;
            int   hitIdx = -1; // no hit

            if (sensorHit && tSensor > 0.f) {
                // So far, sensor is the best candidate intersection
                tMin   = tSensor;
                hitIdx = -2;  // special ID => sensor
            }

            // B) Intersect with scene geometry
            for (size_t i = 0; i < m_gpuData.numGaussians; ++i) {
                if (i == lightIdx && bounce == 0) {
                    // Optionally skip self-intersection on the first bounce
                    // because we don't want to instantly re-hit the emitter patch.
                    continue;
                }

                const glm::vec3& patchPos  = m_gpuData.gaussianInputAssembly[i].position;
                const glm::vec3& patchNorm = m_gpuData.gaussianInputAssembly[i].normal;

                float tPlane = FLT_MAX;
                if (intersectPlane(patchNorm, patchPos, rayOrigin, rayDir, tPlane)) {
                    if (tPlane > 0.f && tPlane < tMin) {
                        tMin   = tPlane;
                        hitIdx = static_cast<int>(i);
                    }
                }
            }

            // If no intersection => photon escapes
            if (hitIdx < 0) {
                return;
            }

            // Check if hit is sensor or a patch
            if (hitIdx == -2) {
                // The photon hits the sensor plane
                glm::vec3 sensorPt = rayOrigin + tMin * rayDir;

                // Accumulate flux => this is *in addition* to what might happen
                // in a typical one-bounce approach. Here, we accumulate partial
                // or full throughput at each sensor intersection.
                accumulateOnSensor(sensorPt, throughput);

                // Usually for a *real* sensor, once you hit it, you're done.
                // But if you're modeling a "sensor as just a patch that can reflect,"
                // you could continue. Typically we do:
                return;
            }

            // We have an intersection with a scene patch
            glm::vec3 hitPoint = rayOrigin + tMin * rayDir;
            const GaussianInputAssembly& hitPatch = m_gpuData.gaussianInputAssembly[hitIdx];

            // -------------------------------------------------
            // C) BRDF sampling => new direction
            // -------------------------------------------------
            // Evaluate the BRDF, pick a new direction
            glm::vec3 bounceDir;
            float brdfVal = evaluateBRDF(hitPatch, rayDir, &bounceDir, photonID);

            // -------------------------------------------------
            // D) Update Throughput
            // -------------------------------------------------
            {
                // Typical form for Lambertian reflection:
                // throughput *= (brdfVal * cosTheta / pdf).
                // For simplicity, assume PDF = cosTheta / π, or whatever is in sampleRandomHemisphereAround()
                float cosIn   = glm::dot(hitPatch.normal, -rayDir);
                float cosOut  = glm::dot(hitPatch.normal,  bounceDir);
                if (cosIn < 0.f)  cosIn  = 0.f;
                if (cosOut < 0.f) cosOut = 0.f;

                // Example: multiply throughput by brdfVal, cosIn, etc.
                // Distance attenuation from old to new is optional here
                // (since we do point-to-patch in your approach).
                throughput *= (brdfVal * cosIn);

                // If you want more physically correct reflection,
                // you also factor in PDF or cosOut, etc.,
                // depending on your sampling strategy.
            }

            // -------------------------------------------------
            // E) Russian Roulette Termination?
            // -------------------------------------------------
            // Probability that we continue
            float rrProb = 0.9f; // e.g. 90% chance to continue
            if (bounce >= 2) {
                // e.g. scale by throughput or make more sophisticated
                // so that we kill paths that are too dim.
                rrProb = glm::clamp(throughput / 10.f, 0.1f, 0.9f);
            }
            float rnd = randomFloat(photonID, bounce);  // your random
            if (rnd > rrProb) {
                // kill this photon
                return;
            } else {
                // if survived, scale throughput to keep estimator unbiased
                throughput /= rrProb;
            }

            // -------------------------------------------------
            // F) Move to next bounce
            // -------------------------------------------------
            rayOrigin = hitPoint;
            rayDir    = bounceDir;
        }
        // Exited the loop => we reached m_maxBounces bounces w/o hitting sensor again
    }

    // ---------------------------------------------------------
    // Intersection with plane
    // ---------------------------------------------------------
    bool intersectPlane(const glm::vec3 &planeNormal,
                        const glm::vec3 &planePos,
                        const glm::vec3 &rayOrigin,
                        const glm::vec3 &rayDir,
                        float &t) const
    {
        float denom = glm::dot(planeNormal, rayDir);
        if (fabs(denom) > 1e-6f) {
            glm::vec3 p0l0 = planePos - rayOrigin;
            t = glm::dot(p0l0, planeNormal) / denom;
            return (t >= 0.f);
        }
        return false;
    }

    // ---------------------------------------------------------
    // Evaluate BRDF & pick bounce direction (same style as before)
    // ---------------------------------------------------------
    float evaluateBRDF(const GaussianInputAssembly& hitPatch,
                       const glm::vec3& incomingDir,
                       glm::vec3* outgoingDir,
                       size_t photonID) const
    {
        // Example: Lambertian
        float rho_d = hitPatch.diffuse;
        *outgoingDir = sampleRandomHemisphereAround(hitPatch.normal, photonID);
        return rho_d / M_PI;
    }

    // ---------------------------------------------------------
    // Accumulate flux on sensor (same style as before)
    // ---------------------------------------------------------
    void accumulateOnSensor(const glm::vec3 &sensorHitPoint, float flux) const
    {
        int sensorPixelX, sensorPixelY;
        projectSensorPoint(sensorHitPoint, sensorPixelX, sensorPixelY);

        if (sensorPixelX >= 0 && sensorPixelX < (int)m_camera.m_width &&
            sensorPixelY >= 0 && sensorPixelY < (int)m_camera.m_height)
        {
            // pseudo-code to add flux
            // size_t idx = (sensorPixelY * m_camera.m_width + sensorPixelX) * 4;
            // m_gpuData.imageAccumBuffer[idx + 0] += flux;
            // ...
        }
    }

    // ---------------------------------------------------------
    // Utility for random
    // ---------------------------------------------------------
    float randomFloat(size_t photonID, size_t bounce) const
    {
        // Very naive example
        // You should use a real PRNG per (photonID,bounce).
        uint64_t seed = (photonID+1)*73856093ULL ^ (bounce+1)*19349663ULL;
        seed ^= (seed >> 13);
        return float(seed % 10000) / 10000.f;
    }

    // ---------------------------------------------------------
    // Randomly pick an emissive patch
    // ---------------------------------------------------------
    size_t sampleRandomEmissivePatch(size_t photonID) const
    {
        // pick patch 0 for demo
        return 0;
    }

    // ---------------------------------------------------------
    // Sample direction from patch emission
    // ---------------------------------------------------------
    glm::vec3 sampleEmissionDirection(size_t photonID, const glm::vec3 &normal) const
    {
        glm::vec3 dir = randomUnitVector(photonID);
        // If you want isotropic, just use dir.
        // If you want hemisphere around normal:
        if (glm::dot(dir, normal) < 0) {
            dir = -dir;
        }
        return glm::normalize(dir);
    }

    // ---------------------------------------------------------
    // Project sensor point to pixel coords
    // ---------------------------------------------------------
    void projectSensorPoint(const glm::vec3 &sensorPt,
                            int &px, int &py) const
    {
        // user-defined
        px = static_cast<int>(sensorPt.x + 100.0f);
        py = static_cast<int>(sensorPt.y + 100.0f);
    }

    // ---------------------------------------------------------
    // Sample random hemisphere
    // ---------------------------------------------------------
    glm::vec3 sampleRandomHemisphereAround(const glm::vec3 &normal,
                                           size_t photonID) const
    {
        glm::vec3 randVec = randomUnitVector(photonID);
        if (glm::dot(randVec, normal) < 0.f) {
            randVec = -randVec;
        }
        return glm::normalize(randVec);
    }

    // ---------------------------------------------------------
    // Dummy random unit vector
    // ---------------------------------------------------------
    glm::vec3 randomUnitVector(size_t photonID) const
    {
        float rx = (float)((photonID * 13431) % 100) / 100.f;
        float ry = (float)((photonID * 72381) % 100) / 100.f;
        float rz = (float)((photonID * 51373) % 100) / 100.f;
        glm::vec3 v(rx - 0.5f, ry - 0.5f, rz - 0.5f);
        float len2 = glm::dot(v,v);
        if (len2 < 1e-8f) {
            return glm::vec3(0.f,1.f,0.f);
        }
        return v / sqrtf(len2);
    }
};


}


#endif //LIGHTTRACERKERNELS_H
