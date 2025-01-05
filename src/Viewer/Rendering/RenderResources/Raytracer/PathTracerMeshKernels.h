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
                              PinholeCamera* camera,
                              uint32_t maxBounces,
                              uint32_t frameID)
            : m_gpuData(gpuData), m_numPhotons(numPhotons), m_cameraTransform(cameraPose), m_camera(camera),
              m_maxBounces(maxBounces), m_frameID(frameID) {
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
        uint32_t m_frameID = 0;

        TransformComponent m_cameraTransform{};
        PinholeCamera* m_camera{};
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
            char* tag = m_gpuData.tagComponents[lightEntityIdx].getTagForKernel();
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

                    const char* entityTag = m_gpuData.tagComponents[entityIdx].getTagForKernel();

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

                        const glm::vec3& aLocal = m_gpuData.vertices[i0].position;
                        const glm::vec3& bLocal = m_gpuData.vertices[i1].position;
                        const glm::vec3& cLocal = m_gpuData.vertices[i2].position;

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

                // (A) Check if the ray passes "through" the pinhole
                bool pinholeHit = false;
                float tPinhole = 0.0f;
                glm::vec3 closePt(0.f);
                float apertureDiameter = (m_camera->m_focalLength / m_camera->m_fNumber) / 1000;
                pinholeHit = checkPinholeIntersection(
                    rayOrigin,
                    rayDir,
                    m_cameraTransform.getPosition(),
                    apertureDiameter / 2,
                    closePt, // out: the closest approach on the ray
                    tPinhole // out: parameter t
                );

                if (pinholeHit) {
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
                        float length = glm::length(hitPointWorld);
                        float xWorld = hitPointWorld.x;
                        float yWorld = hitPointWorld.y;
                        float zWorld = hitPointWorld.z;
                        //sycl::ext::oneapi::experimental::printf("Photon %d, Hit camera at: (%f,%f,%f)\n", photonID, hitPointWorld.x, hitPointWorld.y, hitPointWorld.z);
                        if (tCam < tGeom) {
                            float totalFlux = photonFlux * contribution;
                            if (totalFlux > emissionPower)
                                return;

                            accumulateOnSensor(photonID, hitPointWorld, totalFlux);
                            return; // Photon path terminates
                        }
                        // Photon hits the camera plane first
                    }
                }

                // If no hit or geometry hit first, proceed
                if (hit) {
                    //std::cout << "Photon: " << photonID << "Hit sensor at: (" << px << ", " << py <<") | flux: " << photonFlux << std::endl;
                    //sycl::ext::oneapi::experimental::printf("Photon %d, Hit Entity: %d\n", photonID, hitEntity);
                    // Handle emissive surfaces differently if needed
                    // For example, terminate or continue based on material properties
                    // Here, we assume diffuse (Lambertian) reflection

                    // Retrieve material properties
                    const MaterialComponent& mat = m_gpuData.materials[hitEntity];
                    float albedo = 1.0f; //mat.albedo.x; // Assuming monochrome; extend as needed

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
                    float rnd = randomFloat(photonID * m_frameID, bounce * m_frameID);
                    if (rnd > rrProb) {
                        return; // Photon terminated
                    }
                    else {
                        photonFlux /= rrProb; // Adjust flux to maintain unbiasedness
                    }



                    // Sample new direction (Lambertian reflection)
                    //glm::vec3 newDir = sampleRandomHemisphere(hitNormalWorld, photonID, bounce);
                    rayOrigin = hitPointWorld + hitNormalWorld * 1e-4f; // Offset to prevent self-intersection

                    glm::vec3 newDir = sampleDirectionTowardAperture(
                        rayOrigin,
                        m_cameraTransform.getPosition(), // center of aperture
                        glm::vec3(-1, 0, 0), // might be -X if your camera faces X, or -Z, etc.
                        apertureDiameter * 0.5f,
                        photonID * 999 + m_frameID
                    );

                    newDir = mix(sampleRandomHemisphere(hitNormalWorld, photonID, bounce), newDir, 0.3f); // Blend uniform sampling with light direction

                    if (glm::dot(newDir, hitNormalWorld) < 0.f) {
                        newDir = -newDir;
                    }

                    rayDir = glm::normalize(newDir);
                }
                else {
                    // No hit; photon escapes the scene
                    return;
                }
            } // end for bounces

            // If we exit here, we used up all bounces w/o hitting sensor
        }

        glm::vec3 samplePointOnDisk(uint64_t seed,
                            const glm::vec3& center,
                            const glm::vec3& normal,
                            float radius) const
        {
            // Or use any 2D disk sampling approach (e.g., concentric disk sampling).
            // We'll do a simple naive approach:
            float r = radius * sqrt(randomFloat(seed, 0));
            float theta = 2.f * M_PI * randomFloat(seed, 1);

            // Construct orthonormal basis for the disk plane
            glm::vec3 u = normalize(cross(normal, glm::vec3(0.0f, 0.0f, 1.0f))); // pick any stable "someOtherVec"
            glm::vec3 v = cross(normal, u);

            float dx = r * cos(theta);
            float dy = r * sin(theta);

            glm::vec3 offset = dx * u + dy * v;
            return center + offset;
        }

        glm::vec3 sampleDirectionTowardAperture(
            const glm::vec3& lightPos,
            const glm::vec3& apertureCenter,
            const glm::vec3& apertureNormal,
            float apertureRadius,
            uint64_t seed) const
        {
            // pick random point on the lens
            glm::vec3 lensPoint = samplePointOnDisk(seed, apertureCenter, apertureNormal, apertureRadius);

            // direction from light to lens point
            glm::vec3 dir = lensPoint - lightPos;
            return normalize(dir);
        }

        bool checkCameraPlaneIntersection(
            const glm::vec3& rayOriginWorld,
            const glm::vec3& rayDirWorld,
            glm::vec3& hitPointCam, // out: intersection in camera space
            float& tIntersect, // out: parameter t
            float& contributionScore // out: parameter contributionScore
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

        // ---------------------------------------------------------------------
        //  accumulateOnSensor
        // ---------------------------------------------------------------------
        void accumulateOnSensor(size_t photonID, const glm::vec3& hitPointWorld, float photonFlux) const {
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
                return; // Behind the camera; discard
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

                float contribution = photonFlux;
                // Atomic addition for imageMemory
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomicImageMemory(
                    m_gpuData.imageMemory[pixelIndex]);
                atomicImageMemory.fetch_add(contribution);
                // Atomic increment for photonsAccumulated
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> atomicPhotonsAccumulated(
                    m_gpuData.renderInformation->photonsAccumulated);
                atomicPhotonsAccumulated.fetch_add(1);
            }
        }


        // ---------------------------------------------------------
        //  checkPinholeIntersection
        // ---------------------------------------------------------
        bool checkPinholeIntersection(const glm::vec3& rayOrigin,
                                      const glm::vec3& rayDir,
                                      const glm::vec3& pinholeCenter,
                                      float pinholeRadius,
                                      glm::vec3& outClosestPt,
                                      float& outT) const {
            glm::vec3 pinholeNormal = glm::vec3(-1, 0, 0);
            return intersectDisk(pinholeNormal, pinholeCenter, pinholeRadius, rayOrigin, rayDir, outClosestPt, outT);


            /*
            // Step 1: find t* that minimizes distance from line to pinholeCenter
            glm::vec3 OP = rayOrigin - pinholeCenter;
            float denom = glm::dot(rayDir, rayDir); // typically 1 if rayDir is normalized
            if (denom < 1e-14f) {
                return false; // degenerately small direction
            }

            float tStar = -glm::dot(OP, rayDir) / denom;
            if (tStar < 0.0f) {
                return false; // intersection "behind" the emitter
            }

            // Step 2: compute the actual closest point
            outClosestPt = rayOrigin + tStar * rayDir;
            // distance from pinhole center
            glm::vec3 diff = outClosestPt - pinholeCenter;
            float distSq = glm::dot(diff, diff);

            // If dist <= pinholeRadius => "hit"
            if (distSq <= pinholeRadius * pinholeRadius) {
                outT = tStar;
                return true;
            }
            */
            return false;
        }

        // ---------------------------------------------------------------------
        //  Helper: sample an emissive triangle
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveTriangle(size_t photonID, size_t& emissiveEntityIdx) const {
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
                                             glm::vec3& outPos,
                                             glm::vec3& outNormal,
                                             float& emissionPower) const {
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
            const glm::vec3& A = m_gpuData.vertices[i0].position;
            const glm::vec3& B = m_gpuData.vertices[i1].position;
            const glm::vec3& C = m_gpuData.vertices[i2].position;

            const glm::vec3& N_A = m_gpuData.vertices[i0].normal;
            const glm::vec3& N_B = m_gpuData.vertices[i1].normal;
            const glm::vec3& N_C = m_gpuData.vertices[i2].normal;

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
            const MaterialComponent& material = m_gpuData.materials[emissiveEntityIdx];
            emissionPower = material.emission;
        }

        static bool rayTriangleIntersect(
            glm::vec3& ray_origin,
            glm::vec3& ray_dir,
            const glm::vec3& a,
            const glm::vec3& b,
            const glm::vec3& c,
            glm::vec3& out_intersection) {
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
        bool intersectPlane(const glm::vec3& planeNormal,
                            const glm::vec3& planePos,
                            const glm::vec3& rayOrigin,
                            const glm::vec3& rayDir,
                            float& t) const {
            float denom = glm::dot(planeNormal, rayDir);
            if (fabs(denom) > 1e-6f) {
                glm::vec3 p0l0 = planePos - rayOrigin;
                t = glm::dot(p0l0, planeNormal) / denom;
                return (t >= 0.f);
            }
            return false;
        }

        bool intersectDisk(const glm::vec3& normal, const glm::vec3& center, const float& radius,
                           const glm::vec3& rayOrigin, const glm::vec3& rayDir, glm::vec3& outP,
                           float& t) const {
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
        //  sampleEmissionDirection
        // ---------------------------------------------------------------------
        glm::vec3 sampleEmissionDirection(size_t photonID, const glm::vec3& normal) const {
            // Hemisphere around normal or isotropic
            glm::vec3 d = randomUnitVector(photonID * 999 + 1212 * m_frameID); // some PRNG usage
            return glm::normalize(d);
        }


        // ---------------------------------------------------------------------
        //  sampleRandomHemisphere (Lambertian reflection)
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomHemisphere(const glm::vec3& normal,
                                         size_t photonID,
                                         uint32_t bounce) const {
            glm::vec3 r = randomUnitVector(photonID * 23 + bounce * 59 * m_frameID);
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
            auto randomLCG = [](uint64_t& state) -> float {
                state = state * 6364136223846793005ULL + 1; // LCG formula
                return (state >> 33) / float(1ULL << 31); // Normalize to [0, 1)
            };

            uint64_t state = combinedSeed;

            // Random spherical coordinates
            float theta = randomLCG(state) * 2.0f * M_PI; // Random angle [0, 2π]
            float z = randomLCG(state) * 2.0f - 1.0f; // Random z in [-1, 1]
            float r = sqrtf(1.0f - z * z); // Radius of circle at z

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


        struct PCG32 {
            uint64_t state;
            uint64_t inc;

            // Initialize the RNG with a seed and sequence
            void init(uint64_t seed, uint64_t sequence = 1) {
                state = 0;
                inc = (sequence << 1u) | 1u; // Increment must be odd
                nextUInt(); // Advance state
                state += seed;
                nextUInt(); // Advance state again
            }

            // Generate the next uint32_t random number
            uint32_t nextUInt() {
                uint64_t old_state = state;
                state = old_state * 6364136223846793005ULL + inc;
                uint32_t xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
                uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
                return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            }

            // Generate a random float in [0, 1)
            float nextFloat() {
                return (nextUInt() & 0xFFFFFF) / static_cast<float>(1 << 24);
            }
        };

        // ---------------------------------------------------------------------
        //  randomUnitVector using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 randomUnitVector(PCG32& rng) {
            float theta = rng.nextFloat() * 2.0f * M_PI; // [0, 2π)
            float z = rng.nextFloat() * 2.0f - 1.0f; // [-1, 1)
            float r = sqrtf(1.0f - z * z); // Radius at z

            float x = r * cosf(theta);
            float y = r * sinf(theta);

            return glm::vec3(x, y, z); // Already normalized
        }

        // ---------------------------------------------------------------------
        //  sampleEmissionDirection using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleEmissionDirection(const glm::vec3& normal, PCG32& rng) {
            glm::vec3 d = randomUnitVector(rng);
            return glm::normalize(d);
        }

        // ---------------------------------------------------------------------
        //  sampleRandomHemisphere (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomHemisphere(const glm::vec3& normal, PCG32& rng) {
            glm::vec3 r = randomUnitVector(rng);
            if (glm::dot(r, normal) < 0.f) {
                r = -r;
            }
            return glm::normalize(r);
        }
    };
}


#endif //PathTracerMeshKernels
