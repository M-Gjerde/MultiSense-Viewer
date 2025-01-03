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
                              uint32_t width,
                              uint32_t height,
                              uint32_t size,
                              TransformComponent cameraPose,
                              PinholeCamera *camera,
                              uint32_t maxBounces,
                              uint32_t frameID)
                : m_gpuData(gpuData), m_numPhotons(numPhotons), m_width(width), m_height(height), m_size(size),
                  m_cameraTransform(cameraPose), m_camera(camera), m_maxBounces(maxBounces), m_frameID(frameID) {}

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
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_size;
        uint32_t m_maxBounces;  // e.g. 5, 8, or 10
        uint32_t m_frameID;

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
            glm::vec3 rayDir = sampleEmissionDirection(photonID, emitNormalWorld);

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

            // 3) Multi-bounce loop
            for (uint32_t bounce = 0; bounce < m_maxBounces; ++bounce) {
                // A) Intersect with the scene
                float closest_t = FLT_MAX;
                bool hit = false;
                size_t hitEntity = 0;
                size_t hitTri = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);

                // Optionally, also check intersection with a sensor plane first
                // if you want “check sensor before geometry.” Or treat sensor as geometry.
                // We'll treat sensor as geometry for a purely mesh-based approach.

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
                            if (dist < closest_t && dist > 1e-6f) {
                                closest_t = dist;
                                hit = true;
                                hitEntity = entityIdx;
                                hitTri = t;
                                hitPointWorld = worldHit;

                                // compute normal in world space
                                glm::vec3 nLocal = glm::cross(bLocal - aLocal, cLocal - aLocal);
                                glm::vec3 nWorld = glm::mat3(glm::transpose(glm::inverse(entityTransform))) * nLocal;
                                hitNormalWorld = glm::normalize(nWorld);
                            }
                        }
                    }
                }
                // Check for intersection with camera plane:: Direct lighting
                glm::vec3 camHit;
                float tCam;
                float contribution;
                bool cameraHit = checkCameraPlaneIntersection(rayOrigin, rayDir, camHit, tCam, contribution);

                // Determine the closest hit (geometry vs. camera plane)
                if (cameraHit) {
                    // Calculate geometry intersection distance if hit
                    float tGeom = hit ? closest_t : FLT_MAX;
                    glm::vec3 hitPointWorld = rayOrigin + rayDir * tCam;

                    //sycl::ext::oneapi::experimental::printf("Photon %d, Hit camera at: (%f,%f,%f)\n", photonID, hitPointWorld.x, hitPointWorld.y, hitPointWorld.z);
                    if (tCam < tGeom) {
                        accumulateOnSensor(photonID, hitPointWorld, photonFlux * contribution);
                        return; // Photon path terminates
                    }
                    // Photon hits the camera plane first


                }
                return;

                // If no hit or geometry hit first, proceed
                if (hit) {

                    //std::cout << "Photon: " << photonID << "Hit sensor at: (" << px << ", " << py <<") | flux: " << photonFlux << std::endl;
                    //sycl::ext::oneapi::experimental::printf("Photon %d, Hit Entity: %d\n", photonID, hitEntity);
                    // Handle emissive surfaces differently if needed
                    // For example, terminate or continue based on material properties
                    // Here, we assume diffuse (Lambertian) reflection

                    // Retrieve material properties
                    const MaterialComponent &mat = m_gpuData.materials[hitEntity];
                    float albedo = mat.albedo.x; // Assuming monochrome; extend as needed

                    // Calculate cosine of the angle between normal and incoming direction
                    float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                    if (cosTheta < 0.0f) cosTheta = 0.0f;

                    // Update photon flux based on albedo and cosine term
                    photonFlux *= (albedo * cosTheta);

                    // Russian Roulette termination
                    float rrProb = 0.9f; // Base probability
                    if (bounce >= 2) {
                        rrProb = glm::clamp(photonFlux, 0.05f, 0.9f);
                    }
                    float rnd = randomFloat(photonID, bounce);
                    if (rnd > rrProb) {
                        return; // Photon terminated
                    } else {
                        photonFlux /= rrProb; // Adjust flux to maintain unbiasedness
                    }

                    // Sample new direction (Lambertian reflection)
                    glm::vec3 newDir = sampleRandomHemisphere(hitNormalWorld, photonID, bounce);
                    rayOrigin = hitPointWorld + hitNormalWorld * 1e-4f; // Offset to prevent self-intersection
                    rayDir = glm::normalize(newDir);
                } else {
                    // No hit; photon escapes the scene
                    return;
                }
            } // end for bounces

            // If we exit here, we used up all bounces w/o hitting sensor
        }

        // ---------------------------------------------------------------------
        //  Helper: sample an emissive triangle
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveTriangle(size_t photonID, size_t &emissiveEntityIdx) const {
            // Simple Linear Congruential Generator (LCG) for RNG

            auto xorshift64 = [](size_t x) -> size_t {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                return x * 2685821657736338717ULL; // A large prime multiplier
            };

            size_t seed = xorshift64(photonID ^ m_frameID); // Combine photon and frame IDs

            // Iterate through all entities to find emissive ones
            for (size_t entityIdx = 0; entityIdx < m_gpuData.numEntities; ++entityIdx) {
                if (m_gpuData.materials[entityIdx].emission > 0.f) {
                    // Found an emissive entity
                    uint32_t startIndex = m_gpuData.indexOffsets[entityIdx];
                    uint32_t endIndex = (entityIdx + 1 < m_gpuData.numEntities)
                                        ? m_gpuData.indexOffsets[entityIdx + 1]
                                        : m_gpuData.totalIndices;

                    if (endIndex <= startIndex) {
                        continue; // No triangles in this entity
                    }

                    size_t indexCount = endIndex - startIndex;
                    size_t triangleCount = indexCount / 3;

                    if (triangleCount == 0) {
                        continue; // No complete triangles
                    }

                    // Randomly select a triangle index within [0, triangleCount - 1]
                    seed = xorshift64(seed); // Update seed
                    size_t randomTri = seed % triangleCount; // Select a triangle

                    // Calculate the starting index of the selected triangle
                    size_t selectedTriStartIndex = startIndex + (randomTri * 3);

                    // Assign the emissive entity index
                    emissiveEntityIdx = entityIdx;

                    return selectedTriStartIndex;
                }
            }

            // Fallback: No emissive triangles found
            emissiveEntityIdx = 0;
            return 0;
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
            float r1 = randomFloat(photonID, 100);
            float r2 = randomFloat(photonID, 101);
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

        // ---------------------------------------------------------------------
        //  sampleEmissionDirection
        // ---------------------------------------------------------------------
        glm::vec3 sampleEmissionDirection(size_t photonID, const glm::vec3 &normal) const {
            // Hemisphere around normal or isotropic
            glm::vec3 d = randomUnitVector(photonID * 2 + 0); // some PRNG usage
            if (glm::dot(d, normal) < 0.f) {
                d = -d;
            }
            return glm::normalize(d);
        }

        // ---------------------------------------------------------------------
        //  isSensorTriangle
        // ---------------------------------------------------------------------
        bool isSensorTriangle(size_t entityIdx, size_t triIdx) const {
            // If you have a specific entity or material for the sensor, check that
            // Or store a set of “sensorEntities” in GPUData, etc.
            if (m_gpuData.materials[entityIdx].isSensor) {
                return true;
            }
            return false;
        }

        // ---------------------------------------------------------------------
        //  accumulateOnSensor
        // ---------------------------------------------------------------------
        void accumulateOnSensor(size_t photonID, const glm::vec3 &hitPointWorld, float photonFlux) const {
            //
            // 1. Transform the hit point from world space to camera space
            //
            // m_cameraTransform is presumably a component holding the camera's world matrix.
            // We typically need the inverse of that matrix to go from world -> camera space.
            glm::mat4 worldToCamera = glm::inverse(m_cameraTransform.getTransform());
            glm::vec4 hitPointCam = worldToCamera * glm::vec4(hitPointWorld, 1.0f);

            //
            // 2. Project to the image plane using pinhole intrinsics:
            //    X_cam, Y_cam, Z_cam -> x_proj, y_proj
            //
            //    x_proj = (fx * X_cam / Z_cam) + cx
            //    y_proj = (fy * Y_cam / Z_cam) + cy
            //
            // Important: Z_cam should be > 0 for a point in front of the camera.
            //
            float X_cam = hitPointCam.x;
            float Y_cam = hitPointCam.y;
            float Z_cam = hitPointCam.z;

            // Check if Z_cam is negative (the point is in front of the camera)
            if (Z_cam >= 0.0f) {
                return; // Behind the camera; discard
            }

            float x_proj = -((m_camera->m_fx * X_cam) / Z_cam) + m_camera->m_cx;
            float y_proj = -((m_camera->m_fy * Y_cam) / Z_cam) + m_camera->m_cy;

            //
            // 3. Convert to integer pixel coordinates (rounding or flooring)
            //
            // Often, we do a simple int cast or a floor + 0.5f offset,
            // but you can choose what best suits your sampling approach.
            //
            int px = x_proj;
            int py = y_proj;

            //std::cout << "Photon: " << photonID << "Hit sensor at: (" << px << ", " << py <<") | flux: " << photonFlux << std::endl;
            //sycl::ext::oneapi::experimental::printf("Photon %d, Hit camera at: (%f,%f,%f), Hit Sensor at: (%d, %d), Flux: %f, Camera: (%f, %f)\n", photonID, hitPointCam.x, hitPointCam.y, hitPointCam.z, px, py, photonFlux, m_camera->m_width,m_camera->m_height);

            //
            // 4. Check bounds. If inside the image plane, accumulate flux
            //
            if (px >= 0 && px < static_cast<int>(m_camera->m_width) &&
                py >= 0 && py < static_cast<int>(m_camera->m_height)) {
                // Convert 2D coords -> 1D index
                size_t pixelIndex =
                        (static_cast<size_t>(py) * static_cast<size_t>(m_camera->m_width) + static_cast<size_t>(px)) *
                        4;

                uint8_t contribution = photonFlux * 10;
                // Accumulate the flux
                m_gpuData.imageMemory[pixelIndex + 0] += contribution;
                m_gpuData.imageMemory[pixelIndex + 1] += contribution;
                m_gpuData.imageMemory[pixelIndex + 2] += contribution;
                m_gpuData.imageMemory[pixelIndex + 3] = 255;

            }
        }


        // ---------------------------------------------------------------------
        //  projectSensorPoint
        // ---------------------------------------------------------------------
        void projectSensorPoint(const glm::vec3 &pt, int &px, int &py) const {
            // If your sensor is a plane at Z=-1, you can invert the same logic you had in camera code
            // Or if the sensor is another entity, you'd do barycentric or something.
            // For demonstration, do a simple orthographic map:
            px = (int) (pt.x * 100 + 100);
            py = (int) (pt.y * 100 + 100);
        }

        // ---------------------------------------------------------------------
        //  sampleRandomHemisphere (Lambertian reflection)
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomHemisphere(const glm::vec3 &normal,
                                         size_t photonID,
                                         uint32_t bounce) const {
            glm::vec3 r = randomUnitVector(photonID * 23 + bounce * 59);
            if (glm::dot(r, normal) < 0.f) {
                r = -r;
            }
            return glm::normalize(r);
        }

        // ---------------------------------------------------------------------
        //  randomUnitVector
        // ---------------------------------------------------------------------
        glm::vec3 randomUnitVector(uint64_t seed) const {
            // Very naive
            // Combine seed with frameID for better randomness
            uint64_t combinedSeed = (seed ^ (m_frameID << 21)) * 0x9E3779B97F4A7C15ULL; // Bit mixing with large prime

            // Use an LCG or similar RNG for random values
            auto randomLCG = [](uint64_t &state) -> float {
                state = state * 6364136223846793005ULL + 1; // LCG formula
                return (state >> 33) / float(1ULL << 31);   // Normalize to [0, 1)
            };

            uint64_t state = combinedSeed;

            // Random spherical coordinates
            float theta = randomLCG(state) * 2.0f * M_PI; // Random angle [0, 2π]
            float z = randomLCG(state) * 2.0f - 1.0f;     // Random z in [-1, 1]
            float r = sqrtf(1.0f - z * z);                // Radius of circle at z

            // Convert spherical coordinates to Cartesian
            float x = r * cosf(theta);
            float y = r * sinf(theta);

            return glm::vec3(x, y, z);
        }

        uint32_t xorshift32(uint32_t state) const {
            uint32_t x = state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            return x;
        }

        // ---------------------------------------------------------------------
        //  randomFloat
        // ---------------------------------------------------------------------
        float randomFloat(size_t photonID, uint32_t bounce) const {
            uint32_t seed = static_cast<uint32_t>((photonID + 1) * 73856093ULL ^ (bounce + 1) * 19349663ULL);
            seed = xorshift32(seed);
            return static_cast<float>(seed % 10000) / 10000.f;
        }

        /**
         * @brief Returns a random integer index in the range [minIndex, maxIndex].
         *
         * @param photonID A unique identifier for each photon (used as part of the seed).
         * @param bounce   The current bounce (used as part of the seed).
         * @param minIndex The inclusive lower bound of the range.
         * @param maxIndex The inclusive upper bound of the range.
         * @return int     A random integer in the range [minIndex, maxIndex].
         */
        int randomIndex(size_t photonID, uint32_t bounce, int minIndex, int maxIndex) const {
            if (minIndex > maxIndex) {
                std::swap(minIndex, maxIndex);
            }

            uint32_t seed = static_cast<uint32_t>((photonID + 1) * 73856093ULL ^ (bounce + 1) * 19349663ULL);
            seed = xorshift32(seed);

            uint64_t range = static_cast<uint64_t>(maxIndex) - static_cast<uint64_t>(minIndex) + 1ULL;
            uint64_t randInRange = seed % range;
            return static_cast<int>(randInRange + minIndex);
        }

        bool checkCameraPlaneIntersection(
                const glm::vec3 &rayOriginWorld,
                const glm::vec3 &rayDirWorld,
                glm::vec3 &hitPointCam,   // out: intersection in camera space
                float &tIntersect, // out: parameter t
                float &contributionScore// out: parameter contributionScore
        ) const {
            // 1) Transform to camera space

            glm::mat4 entityTransform = m_cameraTransform.getTransform();
            // Camera plane normal in world space
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(
                    glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 cameraPlanePointWorld = glm::vec3(
                    entityTransform * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f)); // A point on the plane

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

            glm::vec3 intersectionCamSpace = glm::vec3(glm::inverse(entityTransform) * glm::vec4(intersectionPoint, 1.0f));

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
            tIntersect = t;     // Distance along the ray to the intersection
            return true;
        }
    };


}


#endif //PathTracerMeshKernels
