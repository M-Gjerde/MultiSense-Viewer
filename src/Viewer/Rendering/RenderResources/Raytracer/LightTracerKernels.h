//
// Created by magnus on 12/27/24.
//

#ifndef LightTracerKernel_H
#define LightTracerKernel_H

#include "Viewer/Rendering/RenderResources/Raytracer/Definitions.h"

namespace VkRender::RT {
    class LightTracerKernel {
    public:
        LightTracerKernel(GPUData gpuData,
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
            size_t entityID = 0;
            size_t gaussianID = sampleRandomEmissiveGaussian(photonID, entityID);

            glm::vec3 emitPosLocal, emitNormalLocal;
            float emissionPower;
            sampleGaussianPositionAndNormal(entityID, gaussianID, photonID, emitPosLocal, emitNormalLocal,
                                            emissionPower);

            // Get the model transform matrix for the emissive entity
            TransformComponent lightEntityTransform = m_gpuData.transforms[entityID];
            char *tag = m_gpuData.tagComponents[entityID].getTagForKernel();
            // Transform the sampled position to world space
            glm::vec3 emitPosWorld = glm::vec3(lightEntityTransform.getTransform() * glm::vec4(emitPosLocal, 1.0f));

            // Correctly transform the normal to world space using the inverse transpose of the model matrix
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(lightEntityTransform.getTransform())));
            glm::vec3 emitNormalWorld = glm::normalize(normalMatrix * emitNormalLocal);

            // 2) Sample emission direction
            //glm::vec3 rayDir = sampleCosineWeightedHemisphere(emitNormalWorld, photonID);
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
            float apertureDiameter = (m_camera->parameters().focalLength / m_camera->parameters().fNumber) / 1000;
            float apertureRadius = apertureDiameter * 0.5f;

            glm::mat4 entityTransform = m_cameraTransform.getTransform();
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(
                    glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));

            glm::vec3 directLightingDir = sampleDirectionTowardAperture(
                    rayOrigin,
                    m_cameraTransform.getPosition(), // center of aperture
                    cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
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
                    bool hit = geometryIntersection2DGS(directLightingOrigin, directLightingDir, hitEntity, closest_t, hitPointWorld, hitNormalWorld);
                    float tGeom = hit ? closest_t : FLT_MAX;
                    if (tCam < tGeom) {
                        glm::vec3 cameraHitPointWorld = directLightingOrigin + directLightingDir * tCam;

                        float d = glm::length(cameraHitPointWorld) - 1.0f;
                        float cosTheta = glm::dot(directLightingDir, glm::vec3(1.0f, 0.0f, 0.0f));
                        float scaleFactor = (M_PIf * apertureRadius * apertureRadius * d * d) / cosTheta;
                        if (cosTheta > 0.1f) {
                            if (accumulateOnSensor(photonID, cameraHitPointWorld, photonFlux * scaleFactor)) {
                                // Atomic increment for photonsAccumulated
                                sycl::atomic_ref<
                                        unsigned long int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                        atomicPhotonsAccumulated(
                                        m_gpuData.renderInformation->photonsAccumulated);

                                atomicPhotonsAccumulated.fetch_add(static_cast<unsigned long int>(1));

                            }
                        }
                    }
                }
            }

            return;
            // 3) Multi-bounce loop
            for (uint32_t bounce = 0; bounce < m_maxBounces; ++bounce) {
                // A) Intersect with the scene
                float closest_t = FLT_MAX;
                size_t hitEntity = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);


                // check intersection with geometry
                bool hit = geometryIntersection2DGS(rayOrigin, rayDir, hitEntity, closest_t, hitPointWorld, hitNormalWorld);

                // Check direct illumination
                // (A) Check if the ray passes directly "through" the pinhole from the emissive patch
                bool pinholeHit = false;
                float tPinhole = 0.0f;
                glm::vec3 closePt(0.f);
                float apertureDiameter = (m_camera->parameters().focalLength / m_camera->parameters().fNumber) / 1000;
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
                                sycl::atomic_ref<
                                        unsigned long int, sycl::memory_order::relaxed, sycl::memory_scope::device>
                                        atomicPhotonsAccumulated(
                                        m_gpuData.renderInformation->photonsAccumulatedDirect);

                                //atomicPhotonsAccumulated.fetch_add(1);
                            }
                            return; // Photon path terminates
                        }
                    }
                }


                // If we hit some geometry then calculate the bounce
                if (hit) {
                    // Fetch material parameters
                    const MaterialComponent &mat = m_gpuData.materials[hitEntity];
                    float albedo = 0.7f; //mat.albedo.x; // Assuming monochrome for simplicity
                    float specular = mat.specular; // Specular coefficient
                    float shininess = mat.phongExponent;
                    float diffusion = mat.diffuse; // Diffuse coefficient
                    // We'll accumulate our contribution here
                    float totalContribution = 0.0f;

                    if (diffusion <= 0.0f) {
                        // ----------------------------------
                        // METALLIC branch (diffuse == 0)
                        // ----------------------------------
                        // For a purely metallic surface, ignore diffuse lighting altogether.

                        // Compute reflection direction
                        glm::vec3 reflectedDir = glm::reflect(rayDir, hitNormalWorld);

                        // Compute specular highlight (cosAlpha)
                        float cosAlpha = glm::dot(glm::normalize(reflectedDir), glm::normalize(-rayDir));
                        cosAlpha = glm::max(0.0f, cosAlpha);

                        // Multiply by albedo if metals have tinted reflection
                        // e.g. specularContribution = specular * albedo * ...
                        float specularContribution = specular * std::pow(cosAlpha, shininess) / M_PIf;

                        // Purely specular => total = specular only
                        totalContribution = specularContribution;
                    } else {
                        // ----------------------------------
                        // NON-METALLIC branch (diffuse > 0)
                        // ----------------------------------

                        // 1) Diffuse contribution
                        // Compute cosTheta for the diffuse term
                        float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                        cosTheta = glm::max(0.0f, cosTheta); // Clamp to 0 to prevent negative contributions
                        float diffuseContribution = diffusion * albedo * cosTheta / M_PIf;

                        // 2) Specular contribution
                        glm::vec3 reflectedDir = glm::reflect(rayDir, hitNormalWorld);
                        float cosAlpha = glm::dot(glm::normalize(reflectedDir), glm::normalize(-rayDir));
                        cosAlpha = glm::max(0.0f, cosAlpha);
                        float specularContribution = specular * std::pow(cosAlpha, shininess) / M_PIf;

                        // 3) Combine (simple sum)
                        float sumUnweighted = diffuseContribution + specularContribution;

                        // 4) Energy conservation / normalization:
                        //    We want to ensure that the total reflection doesn't exceed 1,
                        //    weight diffuse vs. specular so that sum of their "weights" is 1.
                        float sumForWeights = albedo + specular;
                        if (sumForWeights > 0.0f) {
                            float diffuseWeight = albedo / sumForWeights;
                            float specularWeight = specular / sumForWeights;

                            // Weighted sum
                            totalContribution = diffuseWeight * diffuseContribution
                                                + specularWeight * specularContribution;
                        } else {
                            // Fallback if albedo + specular == 0
                            totalContribution = 0.0f;
                        }
                    }

                    // Finally, scale the photonFlux (or outgoing radiance) by total contribution
                    photonFlux *= totalContribution;

                    // Russian Roulette termination
                    float rrProb = photonFlux;
                    float minProbability = 0.2f; // 20%
                    float maxProbability = 0.9f; // 90%
                    rrProb = glm::clamp(rrProb, minProbability, maxProbability);
                    float rnd = m_rng[photonID].nextFloat();
                    if (rnd > rrProb) {
                        return; // Photon terminated i.e. absorbed by the last surface
                    }
                    photonFlux = photonFlux / rrProb;

                    // Sample new direction (Lambertian reflection)
                    glm::vec3 newDir = sampleRandomDirection(hitNormalWorld, photonID);
                    rayOrigin = hitPointWorld + hitNormalWorld * 1e-4f; // Offset to prevent self-intersection
                    rayDir = glm::normalize(newDir);

                    glm::mat4 entityTransform = m_cameraTransform.getTransform();
                    glm::vec3 cameraPlaneNormalWorld = glm::normalize(
                            glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));

                    glm::vec3 contributionRayDir = sampleDirectionTowardAperture(
                            rayOrigin,
                            m_cameraTransform.getPosition(), // center of aperture
                            cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
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
                            bool hit = geometryIntersection2DGS(contributionRayOrigin, contributionRayDir, hitEntity, closest_t, hitPointWorld, hitNormalWorld);

                            float tGeom = hit ? closest_t : FLT_MAX;
                            if (tCam < tGeom) {
                                glm::vec3 cameraHitPointWorld = contributionRayOrigin + contributionRayDir * tCam;
                                float d = glm::length(cameraHitPointWorld) - 1.0f;
                                float cosTheta = glm::dot(contributionRayDir, glm::vec3(1.0f, 0.0f, 0.0f));
                                float scaleFactor = (M_PIf * apertureRadius * apertureRadius * d * d) / cosTheta;
                                if (cosTheta > 0.1f) {
                                    if (accumulateOnSensor(photonID, cameraHitPointWorld, photonFlux * scaleFactor)) {
                                        // Atomic increment for photonsAccumulated

                                        sycl::atomic_ref<
                                                unsigned long int, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device>
                                                atomicPhotonsAccumulated(
                                                m_gpuData.renderInformation->photonsAccumulated);
                                        atomicPhotonsAccumulated.fetch_add(static_cast<unsigned long int>(1));
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


        bool geometryIntersection2DGS(
    const glm::vec3 &rayOrigin,
    const glm::vec3 &rayDir,
    size_t &hitPointIdx,               // Instead of “hitEntity”, this can store which point we hit
    float &closest_t,
    glm::vec3 &hitPointWorld,
    glm::vec3 &hitNormalWorld
) const
{
    bool hit = false;
    float epsilon = 1e-6f;

    for (uint32_t i = 0; i < m_gpuData.numGaussians; ++i)
    {
        const GaussianInputAssembly& gp = m_gpuData.gaussianInputAssembly[i];
        glm::vec3 N = gp.normal;  // plane normal
        float denom = glm::dot(N, rayDir);

        if (fabs(denom) < epsilon) {
            continue; // nearly parallel
        }

        float t = glm::dot((gp.position - rayOrigin), N) / denom;
        if (t < epsilon) {
            continue; // behind the origin
        }

        glm::vec3 p = rayOrigin + t * rayDir;
        float dist = glm::distance(rayOrigin, p);
        if (dist >= closest_t) {
            continue; // not closer
        }

        // Build local tangent basis on the fly:
        glm::vec3 tAxis, bAxis;
        buildTangentBasis(N, tAxis, bAxis);
        glm::vec3 vPlane = p - gp.position;

        // local coordinates (u,v)
        float u = glm::dot(vPlane, tAxis);
        float v = glm::dot(vPlane, bAxis);

        // Evaluate 2D Gaussian in local space
        float sigmaU = gp.scale.x;
        float sigmaV = gp.scale.y;
        float exponent =
           -0.5f * ( (u*u)/(sigmaU*sigmaU) + (v*v)/(sigmaV*sigmaV) );
        float gaussVal = sycl::exp(exponent);

        // simple cutoff
        if (gaussVal < 0.05f) {
            continue;
        }
        // else it's a valid intersection

        closest_t      = dist;
        hitPointIdx       = i;
        hitPointWorld  = p;
        hitNormalWorld = N;
        hit = true;
    }

    return hit;
}

        inline void buildTangentBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B)
        {
            // Pick an arbitrary vector that is not collinear with N
            glm::vec3 tmp = (fabs(N.x) > 0.9f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);

            // T = normalize(cross(tmp, N))
            T = glm::normalize(glm::cross(tmp, N));

            // B = cross(N, T)
            B = glm::cross(N, T);
            // T, B, and N are now mutually orthonormal
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
            float halfW = (m_camera->parameters().width * 0.5f) / m_camera->parameters().fx;
            float halfH = (m_camera->parameters().height * 0.5f) / m_camera->parameters().fy;

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


            float xSensor_mm = m_camera->parameters().focalLength * (Xc_mm / Zc_mm);
            float ySensor_mm = m_camera->parameters().focalLength * (Yc_mm / Zc_mm);

            float xPitchSize = ((m_camera->parameters().focalLength) / m_camera->parameters().fx);
            float yPitchSize = ((m_camera->parameters().focalLength) / m_camera->parameters().fy);

            float xPixel = (xSensor_mm / xPitchSize) + m_camera->parameters().cx;
            float yPixel = (ySensor_mm / yPitchSize) + m_camera->parameters().cy;


            //float xPixel = m_camera->parameters().fx * (Xc_m / Zc_m) + m_camera->parameters().cx;
            //float yPixel = m_camera->parameters().fy * (Yc_m / Zc_m) + m_camera->parameters().cy;


            int px = static_cast<int>(std::round(xPixel));
            int py = static_cast<int>(std::round(yPixel));


            //
            // 4. Check bounds. If inside the image plane, accumulate flux
            //
            if (px >= 0 && px < static_cast<int>(m_camera->parameters().width) &&
                py >= 0 && py < static_cast<int>(m_camera->parameters().height)) {
                // Convert 2D coords -> 1D index
                size_t pixelIndex =
                        static_cast<size_t>(py) * static_cast<size_t>(m_camera->parameters().width) +
                        static_cast<size_t>(px);
                // Prevent saturation

                /*
                float currentValue = m_gpuData.imageMemory[pixelIndex];
                float newValue = std::min(1.0f, currentValue + photonFlux);
                photonFlux = newValue - currentValue; // Update photonFlux for atomic addition
                */

                photonFlux = std::pow(photonFlux, 1.0f / m_gpuData.renderInformation->gamma);

                float currentValue = m_gpuData.imageMemory[pixelIndex];
                float newValue = std::min(1.0f, currentValue + photonFlux);
                photonFlux = newValue - currentValue; // Update photonFlux for atomic addition

                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomicImageMemory(
                        m_gpuData.imageMemory[pixelIndex]);
                atomicImageMemory.fetch_add(photonFlux);
                // Atomic addition for imageMemory





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
        //  Helper: sample an emissive gaussian object
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveGaussian(size_t photonID, size_t& entityID) const {

            // Simple Linear Congruential Generator (LCG) for RNG
            std::array<size_t, 10> samples{}; // TODo max 10 light sources supported currently
            size_t i = 0;
            for (size_t entityIdx = 0; entityIdx < m_gpuData.numGaussians; ++entityIdx) {
                if (m_gpuData.gaussianInputAssembly[entityIdx].emission > 0.f) {
                    samples[i] = entityIdx;
                    i++;
                }
            }
            entityID = 0;
            // Select a random Gaussian with emissive properties
            return samples[m_rng[photonID].nextUInt() % i];
        }

        // ---------------------------------------------------------------------
        //  sampleTrianglePositionAndNormal
        // ---------------------------------------------------------------------
        void sampleGaussianPositionAndNormal(size_t entityID, size_t emissiveEntityIdx,
                                             size_t photonID,
                                             glm::vec3 &outPos,
                                             glm::vec3 &outNormal,
                                             float &emissionPower) const {


            const GaussianInputAssembly &gaussian = m_gpuData.gaussianInputAssembly[emissiveEntityIdx];
            const TransformComponent &transform = m_gpuData.transforms[entityID];
            // If you have a transform for the mesh, apply it here
            // For demonstration, assume identity:

           // TODO get a random position within the Gaussian's influence
            // Apply the transform to the position
            glm::vec4 transformedPosition = transform.getTransform() * glm::vec4(gaussian.position, 1.0f);
            outPos = glm::vec3(transformedPosition);

            // Apply the transform to the normal (ignore translation, so use mat3)
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform.getTransform())));
            outNormal = glm::normalize(normalMatrix * gaussian.normal);
            emissionPower = gaussian.emission;
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


        // Constructs an orthonormal basis (T, B, N) given a normal N.
        inline void buildTangentBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B) const {
            // Any vector not collinear with N will do for "temp"
            glm::vec3 temp = (fabs(N.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);

            T = glm::normalize(glm::cross(temp, N));
            B = glm::cross(N, T);
            // Now T, B, N is an orthonormal basis
        }

        glm::vec3 sampleCosineWeightedHemisphere(
                const glm::vec3 &normal,
                size_t photonID) // random [0,1]
        const {
            // Step 1: Convert to spherical coords for cosine-weighted distribution
            float u1 = m_rng[photonID].nextFloat();
            float u2 = m_rng[photonID].nextFloat();
            float r = std::sqrt(u1);
            float phi = 2.0f * M_PIf * u2; // M_PIf = float version of pi

            // Step 2: Local coordinates (z up)
            float x = r * std::cos(phi);
            float y = r * std::sin(phi);
            float z = std::sqrt(1.0f - u1);

            // Step 3: Build a local orthonormal basis around 'normal'
            glm::vec3 t, b;
            buildTangentBasis(normal, t, b);

            // Step 4: Transform from local [x, y, z] into world space
            glm::vec3 sampleWorld = x * t + y * b + z * normal;
            return glm::normalize(sampleWorld);
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


#endif //LightTracerKernel
