//
// Created by magnus on 12/27/24.
//

#ifndef PathTracerMeshKernels_H
#define PathTracerMeshKernels_H

#include "Viewer/Rendering/RenderResources/Raytracer/Definitions.h"

namespace VkRender::RT {
    class PathTracerMeshKernels {
    public:
        PathTracerMeshKernels(GPUData gpuData,
                              uint32_t numPhotons,
                              TransformComponent cameraPose,
                              PinholeCamera *camera,
                              uint32_t maxBounces,
                              PCG32 *rng)
                : m_gpuData(gpuData), m_numPhotons(numPhotons), m_cameraTransform(cameraPose), m_camera(camera),
                  m_maxBounces(maxBounces), m_rng(rng) {

        }

        void operator()(sycl::item<1> item) const {
            size_t photonID = item.get_linear_id();
            if (photonID >= m_numPhotons) {
                return;
            }
            // Each thread traces one photon.
            traceOnePhoton(photonID);
        }

    private:
        GPUData m_gpuData{};
        uint32_t m_numPhotons{};
        uint32_t m_maxBounces = 5; // e.g. 5, 8, or 10

        PCG32 *m_rng;
        TransformComponent m_cameraTransform{};
        PinholeCamera *m_camera{};

        // ---------------------------------------------------------
        // Single Photon Trace (Multi-Bounce)
        // ---------------------------------------------------------
        void traceOnePhoton(size_t photonID) const {
            // 1) Pick an emissive triangle and sample a random point on it
            size_t lightEntityIdx = 0;
            size_t lightTriIdx = sampleRandomEmissiveTriangle(photonID, lightEntityIdx);

            glm::vec3 emitPosLocal, emitNormalLocal;
            float emissionPower;
            sampleTrianglePositionAndNormal(lightTriIdx, lightEntityIdx, photonID, emitPosLocal, emitNormalLocal,
                                            emissionPower);

            // Get the model transform matrix for the emissive entity
            TransformComponent lightEntityTransform = m_gpuData.transforms[lightEntityIdx];
            char *tag = m_gpuData.tagComponents[lightEntityIdx].getTagForKernel();
            // Transform the sampled position to world space
            glm::vec3 emitPosWorld = glm::vec3(lightEntityTransform.getTransform() * glm::vec4(emitPosLocal, 1.0f));

            // Correctly transform the normal to world space using the inverse transpose of the model matrix
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(lightEntityTransform.getTransform())));
            glm::vec3 emitNormalWorld = glm::normalize(normalMatrix * emitNormalLocal);

            // 2) Sample emission direction
            glm::vec3 rayDir = sampleRandomDirection(emitNormalWorld, photonID);

            // The photon throughput tracks how much flux remains after each bounce
            float photonFlux = emissionPower; // or you can store a color as a vec3 if needed

            // We start in world space with rayOrigin = emitPos
            glm::vec3 rayOrigin = emitPosWorld;

            float Lx = rayOrigin.x;
            float Ly = rayOrigin.y;
            float Lz = rayOrigin.z;

            float Ldx = rayDir.x;
            float Ldy = rayDir.y;
            float Ldz = rayDir.z;
            // Calculate direct lighting
            float apertureDiameter = (m_camera->m_focalLength / m_camera->m_fNumber) / 1000;
            float apertureRadius = apertureDiameter * 0.5f;

            glm::vec3 directLightingDir = sampleDirectionTowardAperture(
                    rayOrigin,
                    m_cameraTransform.getPosition(), // center of aperture
                    glm::vec3(-1, 0, 0), // might be -X if your camera faces X, or -Z, etc.
                    apertureRadius,
                    photonID
            );
            // Check if contribution ray intersects geometry
            glm::vec3 directLightingOrigin = rayOrigin;


            // Create contribution Rays and trace towards the camera
            if (glm::length(directLightingDir) > 0.1) {
                // Trace our contribution ray
                glm::vec3 camHit;
                float tCam;
                float incidentAngle;
                bool cameraHit = checkCameraPlaneIntersection(directLightingOrigin, directLightingDir, camHit,
                                                              tCam, incidentAngle);
                if (cameraHit) {
                    float closest_t = FLT_MAX;
                    size_t hitEntity = 0;
                    glm::vec3 hitPointWorld(0.0f);
                    glm::vec3 hitNormalWorld(0.0f);
                    // check intersection with geometry
                    bool hit = geometryIntersection(lightEntityIdx, directLightingOrigin, directLightingDir,
                                                    hitEntity, closest_t,
                                                    hitPointWorld, hitNormalWorld);
                    float tGeom = hit ? closest_t : FLT_MAX;
                    if (tCam < tGeom) {
                        glm::vec3 cameraHitPointWorld = directLightingOrigin + directLightingDir * tCam;

                        float d = glm::length(cameraHitPointWorld) - 1.0f;
                        float cosTheta = glm::dot(directLightingDir, glm::vec3(1.0f, 0.0f, 0.0f));
                        float scaleFactor = (M_PIf * apertureRadius * apertureRadius * d * d) / cosTheta;
                        if (cosTheta > 0.1f) {

                            if (accumulateOnSensor(photonID, cameraHitPointWorld, photonFlux * scaleFactor)) {
                                // Atomic increment for photonsAccumulated
                                sycl::atomic_ref<unsigned long int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                        atomicPhotonsAccumulated(
                                        m_gpuData.renderInformation->photonsAccumulated);

                                atomicPhotonsAccumulated.fetch_add(1);
                            }
                        }
                    }
                }
            }

            // 3) Multi-bounce loop
            for (uint32_t bounce = 0; bounce < m_maxBounces; ++bounce) {
                // A) Intersect with the scene
                float closest_t = FLT_MAX;
                size_t hitEntity = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);


                // check intersection with geometry
                bool hit = geometryIntersection(lightEntityIdx, rayOrigin, rayDir, hitEntity, closest_t, hitPointWorld,
                                                hitNormalWorld);

                // Check direct illumination
                // (A) Check if the ray passes "through" the pinhole
                bool pinholeHit = false;
                float tPinhole = 0.0f;
                glm::vec3 closePt(0.f);
                float apertureDiameter = (m_camera->m_focalLength / m_camera->m_fNumber) / 1000;
                float apertureRadius = apertureDiameter * 0.5f;
                pinholeHit = checkPinholeIntersection(
                        rayOrigin,
                        rayDir,
                        m_cameraTransform.getPosition(),
                        apertureRadius,
                        closePt, // out: the closest approach on the ray
                        tPinhole // out: parameter t
                );

                if (pinholeHit) {
                    // Check for intersection with camera plane:: Direct lighting
                    glm::vec3 camHit;
                    float tCam;
                    float incidentAngle;
                    bool cameraHit = checkCameraPlaneIntersection(rayOrigin, rayDir, camHit, tCam, incidentAngle);

                    // Determine the closest hit (geometry vs. camera plane)
                    if (cameraHit) {
                        // Calculate geometry intersection distance if hit
                        float tGeom = hit ? closest_t : FLT_MAX;
                        glm::vec3 hitPointWorld = rayOrigin + rayDir * tCam;
                        float length = glm::length(hitPointWorld);
                        float xWorld = hitPointWorld.x;
                        float yWorld = hitPointWorld.y;
                        float zWorld = hitPointWorld.z;
                        //sycl::ext::oneapi::experimental::printf("Photon %d, Hit camera at: (%f,%f,%f)\n", photonID, hitPointWorld.x, hitPointWorld.y, hitPointWorld.z);
                        if (tCam < tGeom) {

                            if (accumulateOnSensor(photonID, hitPointWorld, photonFlux)) {
                                // Atomic increment for photonsAccumulated
                                sycl::atomic_ref<unsigned long int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                        atomicPhotonsAccumulated(
                                        m_gpuData.renderInformation->photonsAccumulatedDirect);

                                atomicPhotonsAccumulated.fetch_add(1);
                            }
                            return; // Photon path terminates
                        }
                    }
                }




                // If no hit or geometry hit first, proceed
                if (hit) {

                    const MaterialComponent &mat = m_gpuData.materials[hitEntity];
                    float albedo = mat.albedo.x; // Assuming monochrome; replace with mat.albedo.x if applicable
                    float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                    if (cosTheta < 0.0f) cosTheta = 0.0f;

                    float diffuseFlux = albedo * 2.0f * cosTheta;
                    //photonFlux *= diffuseFlux;
                    if (bounce == 0)
                        photonFlux *= albedo * cosTheta / (M_PIf * 2);
                    else
                        photonFlux *= albedo * cosTheta / M_PIf;

                    /*
                    // Retrieve material properties
                    const MaterialComponent& mat = m_gpuData.materials[hitEntity];
                    const float& diffuse = mat.diffuse;        // Diffuse color
                    const float& specular = mat.specular;      // Specular color
                    const float phongExponent = mat.phongExponent; // Shininess factor

                    float albedo = 0.4f; // Assuming monochrome; replace with mat.albedo.x if applicable

                    // Calculate cosine of the angle between normal and incoming direction
                    float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                    if (cosTheta < 0.0f) cosTheta = 0.0f;

                    // Compute view vector (toward the sensor or camera)
                    glm::vec3 reflectedDir = glm::reflect(rayDir, hitNormalWorld);

                    // Compute half-vector between view and light directions
                    glm::vec3 halfVector = glm::normalize(reflectedDir - rayDir);

                    // Compute Blinn-Phong terms
                    float specularTerm = glm::dot(hitNormalWorld, halfVector);
                    if (specularTerm > 0.0f) {
                        specularTerm = powf(specularTerm, phongExponent);
                    } else {
                        specularTerm = 0.0f;
                    }
                    */

                    // Update photon flux based on diffuse and specular components
                    //float specularFlux = specular * specularTerm; // Blinn-Phong specular

                    // Combine diffuse and specular contributions
                    // Russian Roulette termination
                    float rrProb = photonFlux * 0.9f;
                    //float minProbability = 0.2f; // 20%
                    //float maxProbability = 0.9f; // 90%
                    //rrProb = glm::clamp(rrProb, minProbability, maxProbability);
                    float rnd = m_rng[photonID].nextFloat();
                    if (rnd > rrProb) {
                        return; // Photon terminated i.e. absorbed by the last surface
                    } else {
                        photonFlux = photonFlux / rrProb;
                    }

                    // Sample new direction (Lambertian reflection)
                    glm::vec3 newDir = sampleRandomHemisphere(hitNormalWorld, photonID);
                    rayOrigin = hitPointWorld + hitNormalWorld * 1e-4f; // Offset to prevent self-intersection
                    rayDir = glm::normalize(newDir);

                    glm::vec3 contributionRayDir = sampleDirectionTowardAperture(
                            rayOrigin,
                            m_cameraTransform.getPosition(), // center of aperture
                            glm::vec3(-1, 0, 0), // might be -X if your camera faces X, or -Z, etc.
                            apertureRadius,
                            photonID
                    );
                    // Check if contribution ray intersects geometry
                    glm::vec3 contributionRayOrigin = rayOrigin;


                    // Create contribution Rays and trace towards the camera
                    if (glm::length(contributionRayDir) > 0.1) {
                        // Trace our contribution ray
                        glm::vec3 camHit;
                        float tCam;
                        float incidentAngle;
                        bool cameraHit = checkCameraPlaneIntersection(contributionRayOrigin, contributionRayDir, camHit,
                                                                      tCam, incidentAngle);
                        if (cameraHit) {
                            float closest_t = FLT_MAX;
                            size_t hitEntity = 0;
                            glm::vec3 hitPointWorld(0.0f);
                            glm::vec3 hitNormalWorld(0.0f);
                            // check intersection with geometry
                            bool hit = geometryIntersection(lightEntityIdx, contributionRayOrigin, contributionRayDir,
                                                            hitEntity, closest_t,
                                                            hitPointWorld, hitNormalWorld);
                            float tGeom = hit ? closest_t : FLT_MAX;
                            if (tCam < tGeom) {
                                glm::vec3 cameraHitPointWorld = contributionRayOrigin + contributionRayDir * tCam;
                                float d = glm::length(cameraHitPointWorld) - 1.0f;
                                float cosTheta = glm::dot(contributionRayDir, glm::vec3(1.0f, 0.0f, 0.0f));
                                float scaleFactor = (M_PIf * apertureRadius * apertureRadius * d * d) / cosTheta;
                                if (cosTheta > 0.1f) {

                                    if (accumulateOnSensor(photonID, cameraHitPointWorld, photonFlux * scaleFactor)) {
                                        // Atomic increment for photonsAccumulated
                                        sycl::atomic_ref<unsigned long int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                                atomicPhotonsAccumulated(
                                                m_gpuData.renderInformation->photonsAccumulated);

                                        atomicPhotonsAccumulated.fetch_add(1);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // No hit; photon escapes the scene
                    return;
                }
            } // end for bounces

            // If we exit here, we used up all bounces w/o hitting sensor
        }

        glm::vec3 samplePointOnDisk(size_t photonID,
                                    const glm::vec3 &center,
                                    const glm::vec3 &normal,
                                    float radius) const {
            // Or use any 2D disk sampling approach (e.g., concentric disk sampling).
            // We'll do a simple naive approach:
            float r = radius * sqrt(m_rng[photonID].nextFloat());
            float theta = 2.f * M_PI * m_rng[photonID].nextFloat();

            // Construct orthonormal basis for the disk plane
            glm::vec3 u = normalize(cross(normal, glm::vec3(0.0f, 0.0f, 1.0f))); // pick any stable "someOtherVec"
            glm::vec3 v = cross(normal, u);

            float dx = r * cos(theta);
            float dy = r * sin(theta);

            glm::vec3 offset = dx * u + dy * v;
            return center + offset;
        }

        glm::vec3 sampleDirectionTowardAperture(
                const glm::vec3 &lightPos,
                const glm::vec3 &apertureCenter,
                const glm::vec3 &apertureNormal,
                float apertureRadius,
                uint64_t photonID) const {
            // pick random point on the lens
            glm::vec3 lensHitPoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
            // direction from light to lens point
            glm::vec3 dir = lensHitPoint - lightPos;
            return normalize(dir);
        }

        bool checkCameraPlaneIntersection(
                const glm::vec3 &rayOriginWorld,
                const glm::vec3 &rayDirWorld,
                glm::vec3 &hitPointCam, // out: intersection in camera space
                float &tIntersect, // out: parameter t
                float &contributionScore // out: parameter contributionScore
        ) const {
            // 1) Transform to camera space

            glm::mat4 entityTransform = m_cameraTransform.getTransform();
            // Camera plane normal in world space
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(
                    glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 cameraPlanePointWorld = glm::vec3(
                    entityTransform *
                    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)); // A point on the plane

            // Ray-plane intersection

            // Ray-plane intersection calculation
            float denom = glm::dot(cameraPlaneNormalWorld, rayDirWorld);
            if (std::abs(denom) < 1e-6) {
                return false; // Ray is parallel to the plane
            }
            glm::vec3 p0l0 = cameraPlanePointWorld - rayOriginWorld;
            float t = glm::dot(p0l0, cameraPlaneNormalWorld) / denom;
            if (t < 1e-6) {
                return false; // Intersection is behind the ray origin
            }
            glm::vec3 intersectionPoint = rayOriginWorld + t * rayDirWorld;

            float hitx = intersectionPoint.x;
            float hity = intersectionPoint.y;
            float hitz = intersectionPoint.z;

            glm::vec3 intersectionCamSpace = glm::vec3(
                    glm::inverse(entityTransform) * glm::vec4(intersectionPoint, 1.0f));

            float hitCx = intersectionCamSpace.x;
            float hitCy = intersectionCamSpace.y;
            float hitCz = intersectionCamSpace.z;

            // Sensor plane bounds in camera space
            float halfW = (m_camera->m_width * 0.5f) / m_camera->m_fx;
            float halfH = (m_camera->m_height * 0.5f) / m_camera->m_fy;

            // Check bounds
            if (intersectionCamSpace.x < -halfW || intersectionCamSpace.x > halfW ||
                intersectionCamSpace.y < -halfH || intersectionCamSpace.y > halfH) {
                return false; // Outside sensor bounds
            }

            float cosAngle = std::abs(glm::dot(cameraPlaneNormalWorld, rayDirWorld)); // Ensure positive cosine
            contributionScore = cosAngle; // Higher cosine means closer to perpendicular


            // If we reach here, the ray intersects the camera plane within bounds
            hitPointCam = intersectionPoint; // Intersection point in camera space
            tIntersect = t; // Distance along the ray to the intersection
            return true;
        }

        bool geometryIntersection(size_t lightEntityIdx, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir,
                                  size_t &hitEntity, float &closest_t, glm::vec3 &hitPointWorld,
                                  glm::vec3 &hitNormalWorld) const {
            bool hit = false;
            for (uint32_t entityIdx = 0; entityIdx < m_gpuData.numEntities; ++entityIdx) {
                // Transform ray to local space
                if (entityIdx == lightEntityIdx)
                    continue;

                const char *entityTag = m_gpuData.tagComponents[entityIdx].getTagForKernel();

                glm::mat4 entityTransform = m_gpuData.transforms[entityIdx].getTransform();
                glm::mat4 invEntityTransform = glm::inverse(entityTransform);

                glm::vec3 localRayOrigin = glm::vec3(invEntityTransform * glm::vec4(rayOrigin, 1.0f));
                glm::vec3 localRayDir = glm::normalize(glm::vec3(invEntityTransform * glm::vec4(rayDir, 0.0f)));

                // Figure out index range for this entity
                uint32_t startIndex = m_gpuData.indexOffsets[entityIdx];
                uint32_t endIndex = (entityIdx + 1 < m_gpuData.numEntities)
                                    ? m_gpuData.indexOffsets[entityIdx + 1]
                                    : m_gpuData.totalIndices;

                if (endIndex <= startIndex) {
                    continue; // no triangles
                }

                size_t entityIndexCount = endIndex - startIndex;
                size_t triangleCount = entityIndexCount / 3;

                // For each triangle in this entity
                for (size_t t = 0; t < triangleCount; ++t) {
                    uint32_t i0 = m_gpuData.indices[startIndex + t * 3 + 0];
                    uint32_t i1 = m_gpuData.indices[startIndex + t * 3 + 1];
                    uint32_t i2 = m_gpuData.indices[startIndex + t * 3 + 2];

                    const glm::vec3 &aLocal = m_gpuData.vertices[i0].position;
                    const glm::vec3 &bLocal = m_gpuData.vertices[i1].position;
                    const glm::vec3 &cLocal = m_gpuData.vertices[i2].position;

                    glm::vec3 localHit(0.f);
                    if (rayTriangleIntersect(localRayOrigin, localRayDir, aLocal, bLocal, cLocal, localHit)) {
                        glm::vec3 worldHit = glm::vec3(entityTransform * glm::vec4(localHit, 1.0f));
                        float dist = glm::distance(rayOrigin, worldHit);
                        if (dist < closest_t && dist > 1e-3f) {
                            closest_t = dist;
                            hitEntity = entityIdx;
                            hitPointWorld = worldHit;
                            hit = true;
                            // compute normal in world space
                            glm::vec3 nLocal = glm::cross(bLocal - aLocal, cLocal - aLocal);
                            glm::vec3 nWorld = glm::mat3(glm::transpose(glm::inverse(entityTransform))) * nLocal;
                            hitNormalWorld = glm::normalize(nWorld);
                        }
                    }
                }
            }
            return hit;
        }

        // ---------------------------------------------------------------------
        //  accumulateOnSensor
        // ---------------------------------------------------------------------
        bool accumulateOnSensor(size_t photonID, const glm::vec3 &hitPointWorld, float photonFlux) const {
            //
            // 1. Transform the hit point from world space to camera space
            //
            // m_cameraTransform is presumably a component holding the camera's world matrix.
            // We typically need the inverse of that matrix to go from world -> camera space.
            glm::mat4 worldToCamera = glm::inverse(m_cameraTransform.getTransform());
            glm::vec4 hitPointCam = worldToCamera * glm::vec4(hitPointWorld, 1.0f);
            float xWorld = hitPointWorld.x;
            float yWorld = hitPointWorld.y;
            float zWorld = hitPointWorld.z;

            // 2. Project to the image plane using pinhole intrinsics:
            // Important: Z_cam should be > 0 for a point in front of the camera.
            //
            float Xc_m = hitPointCam.x;
            float Yc_m = hitPointCam.y;
            float Zc_m = hitPointCam.z;

            // Convert to mm:
            float Xc_mm = Xc_m * 1000.0f;
            float Yc_mm = Yc_m * 1000.0f;
            float Zc_mm = Zc_m * 1000.0f; // If needed for consistent usage

            if (Zc_m <= 0.0f) {
                return false; // Behind the camera
            }


            float xSensor_mm = m_camera->m_focalLength * (Xc_mm / Zc_mm);
            float ySensor_mm = m_camera->m_focalLength * (Yc_mm / Zc_mm);

            float xPitchSize = ((m_camera->m_focalLength) / m_camera->m_fx);
            float yPitchSize = ((m_camera->m_focalLength) / m_camera->m_fy);

            float xPixel = (xSensor_mm / xPitchSize) + m_camera->m_cx;
            float yPixel = (ySensor_mm / yPitchSize) + m_camera->m_cy;


            //float xPixel = m_camera->m_fx * (Xc_m / Zc_m) + m_camera->m_cx;
            //float yPixel = m_camera->m_fy * (Yc_m / Zc_m) + m_camera->m_cy;


            int px = static_cast<int>(std::round(xPixel));
            int py = static_cast<int>(std::round(yPixel));


            //
            // 4. Check bounds. If inside the image plane, accumulate flux
            //
            if (px >= 0 && px < static_cast<int>(m_camera->m_width) &&
                py >= 0 && py < static_cast<int>(m_camera->m_height)) {
                // Convert 2D coords -> 1D index
                size_t pixelIndex =
                        static_cast<size_t>(py) * static_cast<size_t>(m_camera->m_width) + static_cast<size_t>(px);

                float gamma = 2.2f; // Gamma value > 1 brightens dark areas
                photonFlux = std::pow(photonFlux, 1.0f / gamma);

                // Atomic addition for imageMemory
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomicImageMemory(
                        m_gpuData.imageMemory[pixelIndex]);
                atomicImageMemory.fetch_add(photonFlux);

                return true;
            }
            return false;
        }


        // ---------------------------------------------------------
        //  checkPinholeIntersection
        // ---------------------------------------------------------
        bool checkPinholeIntersection(const glm::vec3 &rayOrigin,
                                      const glm::vec3 &rayDir,
                                      const glm::vec3 &pinholeCenter,
                                      float pinholeRadius,
                                      glm::vec3 &outClosestPt,
                                      float &outT) const {
            glm::vec3 pinholeNormal = glm::vec3(-1, 0, 0);
            return intersectDisk(pinholeNormal, pinholeCenter, pinholeRadius, rayOrigin, rayDir, outClosestPt, outT);

            return false;
        }

        // ---------------------------------------------------------------------
        //  Helper: sample an emissive triangle
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveTriangle(size_t photonID, size_t &emissiveEntityIdx) const {
            // Simple Linear Congruential Generator (LCG) for RNG


            std::array<size_t, 10> samples{}; // TODo max 10 light sources supported currently
            size_t i = 0;
            for (size_t entityIdx = 0; entityIdx < m_gpuData.numEntities; ++entityIdx) {
                if (m_gpuData.materials[entityIdx].emission > 0.f) {
                    samples[i] = entityIdx;
                    i++;
                }
            }

            size_t entityIdx = samples[m_rng[photonID].nextUInt() % i];
            // It
            // erate through all entities to find emissive ones
            // Found an emissive entity
            uint32_t startIndex = m_gpuData.indexOffsets[entityIdx];
            uint32_t endIndex = (entityIdx + 1 < m_gpuData.numEntities)
                                ? m_gpuData.indexOffsets[entityIdx + 1]
                                : m_gpuData.totalIndices;


            if (endIndex <= startIndex) {
                return 0; // No triangles in this entity
            }

            size_t indexCount = endIndex - startIndex;
            size_t triangleCount = indexCount / 3;

            if (triangleCount == 0) {
                return 0; // No complete triangles
            }

            // Randomly select a triangle index within [0, triangleCount - 1]
            size_t randomTri = m_rng[photonID].nextUInt() % triangleCount; // Select a triangle

            // Calculate the starting index of the selected triangle
            size_t selectedTriStartIndex = startIndex + (randomTri * 3);

            // Assign the emissive entity index
            emissiveEntityIdx = entityIdx;

            return selectedTriStartIndex;
        }

        // ---------------------------------------------------------------------
        //  sampleTrianglePositionAndNormal
        // ---------------------------------------------------------------------
        void sampleTrianglePositionAndNormal(size_t triIndex,
                                             size_t emissiveEntityIdx,
                                             size_t photonID,
                                             glm::vec3 &outPos,
                                             glm::vec3 &outNormal,
                                             float &emissionPower) const {
            // triIndex is the starting index of the triangle in the indices array (must be a multiple of 3)
            if (triIndex + 2 >= m_gpuData.totalIndices) {
                // Fallback to default values
                outPos = glm::vec3(0.0f);
                outNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                emissionPower = 0.0f;
                return;
            }

            uint32_t i0 = m_gpuData.indices[triIndex + 0];
            uint32_t i1 = m_gpuData.indices[triIndex + 1];
            uint32_t i2 = m_gpuData.indices[triIndex + 2];


            // For simplicity, assume there's a single entity transform or identity:
            // (In reality, you'd figure out which entity this tri belongs to, apply transform, etc.)
            // Retrieve Triangle Vertices and Their Normals
            const glm::vec3 &A = m_gpuData.vertices[i0].position;
            const glm::vec3 &B = m_gpuData.vertices[i1].position;
            const glm::vec3 &C = m_gpuData.vertices[i2].position;

            const glm::vec3 &N_A = m_gpuData.vertices[i0].normal;
            const glm::vec3 &N_B = m_gpuData.vertices[i1].normal;
            const glm::vec3 &N_C = m_gpuData.vertices[i2].normal;

            // Barycentric Sample for a Random Point
            float r1 = m_rng[photonID].nextFloat();
            float r2 = m_rng[photonID].nextFloat();
            if (r1 + r2 > 1.f) {
                r1 = 1.f - r1;
                r2 = 1.f - r2;
            }
            float u = r1;
            float v = r2;
            float w = 1.0f - u - v;

            glm::vec3 localPos = A + (B - A) * u + (C - A) * v;

            // Interpolate the Normals
            glm::vec3 interpolatedNormal = glm::normalize(N_A * w + N_B * u + N_C * v);

            // If you have a transform for the mesh, apply it here
            // For demonstration, assume identity:
            outPos = localPos;
            outNormal = interpolatedNormal;

            // Derive Emission Power from Material
            const MaterialComponent &material = m_gpuData.materials[emissiveEntityIdx];
            emissionPower = material.emission;
            if (emissionPower > 0.1f)
                int interesting = 1;
        }

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

        // ---------------------------------------------------------
        // Intersection with plane
        // ---------------------------------------------------------
        bool intersectPlane(const glm::vec3 &planeNormal,
                            const glm::vec3 &planePos,
                            const glm::vec3 &rayOrigin,
                            const glm::vec3 &rayDir,
                            float &t) const {
            float denom = glm::dot(planeNormal, rayDir);
            if (fabs(denom) > 1e-6f) {
                glm::vec3 p0l0 = planePos - rayOrigin;
                t = glm::dot(p0l0, planeNormal) / denom;
                return (t >= 0.f);
            }
            return false;
        }

        bool intersectDisk(const glm::vec3 &normal, const glm::vec3 &center, const float &radius,
                           const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, glm::vec3 &outP,
                           float &t) const {
            if (intersectPlane(normal, center, rayOrigin, rayDir, t)) {
                outP = rayOrigin + rayDir * t; // Calculate intersection point
                glm::vec3 v = outP - center; // Vector from disk center to intersection point
                float d2 = glm::dot(v, v); // Squared distance from disk center to intersection point
                // return (sqrtf(d2) <= radius); // Direct distance comparison (less efficient)
                // The optimized method (using precomputed radius squared):
                return d2 <= radius * radius; // Compare squared distances (more efficient)
            }

            return false;
        }


        // ---------------------------------------------------------------------
        //  randomUnitVector using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 randomUnitVector(size_t photonID) const {
            float theta = m_rng[photonID].nextFloat() * 2.0f * M_PI; // [0, 2π)
            float z = m_rng[photonID].nextFloat() * 2.0f - 1.0f; // [-1, 1)
            float r = sqrtf(1.0f - z * z); // Radius at z

            float x = r * cosf(theta);
            float y = r * sinf(theta);

            return glm::vec3(x, y, z); // Already normalized
        }


        // ---------------------------------------------------------------------
        //  sampleRandomHemisphere (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomHemisphere(const glm::vec3 &normal, size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            if (glm::dot(r, normal) < 0.f) {
                r = -r;
            }
            return glm::normalize(r);
        }

        // ---------------------------------------------------------------------
        //  sampleRandomDirection (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomDirection(const glm::vec3 &normal, size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            return glm::normalize(r);
        }
    };
}


#endif //PathTracerMeshKernels
