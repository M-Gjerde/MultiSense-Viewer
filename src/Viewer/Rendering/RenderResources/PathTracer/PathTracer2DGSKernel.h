//
// Created by magnus on 12/27/24.
//

#ifndef LightTracerKernel_H
#define LightTracerKernel_H

#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"

namespace VkRender::PathTracer {
    class LightTracerKernel {
    public:
        LightTracerKernel(GPUData gpuData,
                          GPUDataOutput *gpuDataOutput,
                          PCG32 *rng)
            : m_gpuData(gpuData), m_gpuDataOutput(gpuDataOutput), m_rng(rng) {
            m_cameraTransform = m_gpuData.cameraTransform;
            m_camera = m_gpuData.pinholeCamera;
        }

        void operator()(sycl::item<1> item) const {
            size_t photonID = item.get_linear_id();
            if (photonID >= m_gpuData.renderInformation->totalPhotons) {
                return;
            }
            // Each thread traces one photon.
            traceOnePhoton(photonID);
        }

    private:
        GPUData m_gpuData{};
        GPUDataOutput *m_gpuDataOutput{};

        PCG32 *m_rng;
        TransformComponent *m_cameraTransform{};
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

            // 2) Sample emission direction
            glm::vec3 rayDir = sampleCosineWeightedHemisphere(emitNormalLocal, photonID);
            float photonFlux = emissionPower; // or you can store a color as a vec3 if needed
            glm::vec3 rayOrigin = emitPosLocal;
            float apertureDiameter = (m_camera->parameters().focalLength / m_camera->parameters().fNumber) / 1000;
            // float apertureRadius = apertureDiameter * 0.5f;
            float apertureRadius = 0.00001f;

            glm::mat4 entityTransform = m_cameraTransform->getTransform();
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(
                glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));

            castContributionRay(rayOrigin, cameraPlaneNormalWorld, apertureRadius, photonID, gaussianID, photonFlux);
            // 3) Multi-bounce loop
            for (uint32_t bounce = 0; bounce < m_gpuData.renderInformation->numBounces; ++bounce) {
                // A) Intersect with the scene
                float closest_t = FLT_MAX;
                size_t hitEntity = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);


                // check intersection with geometry
                bool hit = geometryIntersection2DGS(gaussianID, rayOrigin, rayDir, hitEntity, closest_t, hitPointWorld,
                                                    hitNormalWorld);

                // If we hit some geometry then calculate the bounce
                if (hit) {
                    // Fetch material parameters
                    const GaussianInputAssembly &mat = m_gpuData.gaussianInputAssembly[hitEntity];
                    float albedo = mat.color.x;
                    float specular = mat.specular; // Specular coefficient
                    float shininess = mat.phongExponent;
                    float diffuse = mat.diffuse; // Diffuse coefficient
                    // We'll accumulate our contribution here
                    float totalContribution = 1.0f;
                    float contributionRayContribution = 0.0f;

                    // ----------------------------------
                    // NON-METALLIC branch (diffuse > 0)
                    // ----------------------------------
                    // 1) Diffuse contribution
                    // Compute cosTheta for the diffuse term

                    // 2) Specular contribution
                    // TODO also calculate the contribution if I am sampling directly towards the camera instead of just reflecting along the surface normal
                    // Contribution Dir

                    glm::vec3 apertureHitPoint;
                    glm::vec3 contributionRayDir = sampleDirectionTowardAperture(
                        hitPointWorld,
                        m_cameraTransform->getPosition(), // center of aperture
                        cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
                        apertureHitPoint,
                        apertureRadius,
                        photonID
                    );

                    float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                    cosTheta = glm::max(0.0f, cosTheta); // Clamp to 0 to prevent negative contributions
                    float diffuseContribution = cosTheta * diffuse * albedo / M_PIf;


                    glm::vec3 delta = rayDir + contributionRayDir;
                    glm::vec3 halfVector = delta / glm::length(delta);
                    float cosAlpha = std::max(glm::dot(hitNormalWorld, halfVector), 0.0f);
                    float specularContribution = specular * std::pow(cosAlpha, shininess) / M_PIf;

                    float sumForWeights = albedo + specular;
                    if (sumForWeights > 0.0f) {
                        float diffuseWeight = albedo / sumForWeights;
                        float specularWeight = specular / sumForWeights;
                        // Weighted sum
                        contributionRayContribution = diffuseWeight * diffuseContribution
                                                      + specularWeight * specularContribution;
                    }

                    /*
                    // Path contribution
                    {
                        float cosTheta = glm::dot(hitNormalWorld, -rayDir);
                        cosTheta = glm::max(0.0f, cosTheta); // Clamp to 0 to prevent negative contributions
                        float diffuseContribution = diffuse * albedo * cosTheta / M_PIf;

                        glm::vec3 reflectedDir = glm::reflect(rayDir, hitNormalWorld);
                        float cosAlpha = glm::dot(glm::normalize(reflectedDir), glm::normalize(-rayDir));
                        cosAlpha = glm::max(0.0f, cosAlpha);
                        float specularContribution = specular * std::pow(cosAlpha, shininess) / M_PIf;
                        //3 ) Energy conservation / normalization:
                        //    We want to ensure that the total reflection doesn't exceed 1,
                        //    weight diffuse vs. specular so that sum of their "weights" is 1.
                        float sumForWeights = diffuse + specular;
                        if (sumForWeights > 0.0f) {
                            float diffuseWeight = diffuse / sumForWeights;
                            float specularWeight = specular / sumForWeights;
                            // Weighted sum
                            totalContribution = diffuseWeight * diffuseContribution
                                                + specularWeight * specularContribution;
                        } else {
                            // Fallback if albedo + specular == 0
                            totalContribution = 0.0f;
                        }
                    }
                    */

                    float contributionFlux = photonFlux * contributionRayContribution;
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
                    glm::vec3 newDir = sampleCosineWeightedHemisphere(hitNormalWorld, photonID);
                    //glm::vec3 newDir = sampleRandomDirection(photonID);
                    rayOrigin = hitPointWorld + hitNormalWorld * 1e-4f; // Offset to prevent self-intersection
                    rayDir = glm::normalize(newDir);

                    castContributionRay(rayOrigin, cameraPlaneNormalWorld, apertureRadius, photonID, gaussianID,
                                        contributionFlux);
                } else {
                    return;
                }
            } // end for bounces

            // If we exit here, we used up all bounces w/o hitting sensor
        }

        void castContributionRay(const glm::vec3 &rayOrigin, const glm::vec3 &cameraPlaneNormalWorld,
                                 float apertureRadius, size_t photonID, size_t gaussianID, float photonFlux) const {
            // Calculate direct lighting

            glm::vec3 apertureHitPoint;

            glm::vec3 directLightingDir = sampleDirectionTowardAperture(
                rayOrigin,
                m_cameraTransform->getPosition(), // center of aperture
                cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
                apertureHitPoint,
                apertureRadius,
                photonID
            );
            // Early exist if we are hitting the camera plane from behind, this happens if the aperture direction and camera plane normal are parallell or within that quadrant
            float direction = glm::dot(directLightingDir, cameraPlaneNormalWorld);
            if (direction >= 0.0f) {
                return;
            }
            // Check if contribution ray intersects geometry
            glm::vec3 directLightingOrigin = rayOrigin;

            // Create contribution Rays and trace towards the camera
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
                bool hit = geometryIntersection2DGS(gaussianID, directLightingOrigin, directLightingDir, hitEntity,
                                                    closest_t, hitPointWorld, hitNormalWorld);
                float tGeom = hit ? closest_t : FLT_MAX;
                if (tCam < tGeom) {
                    glm::vec3 cameraHitPointWorld = directLightingOrigin + directLightingDir * tCam;

                    float d = glm::length(cameraHitPointWorld);
                    float cosTheta = glm::dot(directLightingDir, -cameraPlaneNormalWorld);
                    //float scaleFactor = (M_PIf * apertureRadius * apertureRadius * d * d) / glm::max(0.1f, cosTheta);
                    float scaleFactor = (M_PIf * 0.000001f * d * d) / glm::max(0.1f, cosTheta);

                    //
                    // 1. Transform the hit point from world space to camera space
                    glm::mat4 worldToCamera = glm::inverse(m_cameraTransform->getTransform());
                    glm::vec4 hitPointCam4 = worldToCamera * glm::vec4(cameraHitPointWorld, 1.0f);
                    glm::vec3 hitPointCam = hitPointCam4 / hitPointCam4.w;

                    accumulateOnSensor(photonID, hitPointCam, photonFlux * scaleFactor);

                    m_gpuDataOutput[photonID].gaussianID = gaussianID;
                    m_gpuDataOutput[photonID].emissionOrigin = rayOrigin;
                    m_gpuDataOutput[photonID].emissionDirection = directLightingDir;
                    m_gpuDataOutput[photonID].hitCamera = cameraHit;
                    m_gpuDataOutput[photonID].emissionDirectionLength = tCam;
                    m_gpuDataOutput[photonID].apertureHitPoint = apertureHitPoint;
                    m_gpuDataOutput[photonID].cameraHitPointLocal = hitPointCam;
                }
            }
        }


        bool geometryIntersection2DGS(
            size_t gaussianID,
            const glm::vec3 &rayOrigin,
            const glm::vec3 &rayDir,
            size_t &hitPointIdx,
            float &closest_t,
            glm::vec3 &hitPointWorld,
            glm::vec3 &hitNormalWorld
        ) const {
            bool hit = false;
            float epsilon = 1e-6f;

            for (uint32_t i = 0; i < m_gpuData.numGaussians; ++i) {
                if (i == gaussianID)
                    continue;

                const GaussianInputAssembly &gp = m_gpuData.gaussianInputAssembly[i];
                glm::vec3 N = gp.normal; // plane normal
                float denom = glm::dot(N, rayDir);

                if (fabs(denom) < epsilon) {
                    continue; // nearly parallel => no valid intersection
                }

                float t = glm::dot((gp.position - rayOrigin), N) / denom;
                if (t < epsilon) {
                    continue; // intersection behind the origin
                }

                glm::vec3 p = rayOrigin + t * rayDir;
                float dist = glm::distance(rayOrigin, p);
                if (dist >= closest_t) {
                    continue; // not closer than current intersection
                }

                // Build local tangent basis for the plane
                glm::vec3 tAxis, bAxis;
                buildTangentBasis(N, tAxis, bAxis);

                // Vector in plane coords
                glm::vec3 vPlane = p - gp.position;
                float u = glm::dot(vPlane, tAxis);
                float v = glm::dot(vPlane, bAxis);

                // Check elliptical boundary:
                float sigmaU = gp.scale.x;
                float sigmaV = gp.scale.y;
                float ellipseParam = (u * u) / (sigmaU * sigmaU)
                                     + (v * v) / (sigmaV * sigmaV);

                if (ellipseParam > 1.0f) {
                    // Outside the ellipse => ignore this intersection
                    continue;
                }

                // If we get here, we have a valid intersection
                closest_t = dist;
                hitPointIdx = i;
                hitPointWorld = p;
                hitNormalWorld = N;
                hit = true;
            }

            return hit;
        }


        glm::vec3 sampleDirectionTowardAperture(
            const glm::vec3 &lightPos,
            const glm::vec3 &apertureCenter,
            const glm::vec3 &apertureNormal,
            glm::vec3 &apertureHitpoint,
            float apertureRadius,
            uint64_t photonID) const {
            // pick random point on the lens
            apertureHitpoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
            // direction from light to lens point
            glm::vec3 dir = apertureHitpoint - lightPos;
            return normalize(dir);
        }

        glm::vec3 samplePointOnDisk(size_t photonID,
                                    const glm::vec3 &center,
                                    const glm::vec3 &normal,
                                    float radius) const {
            // Or use any 2D disk sampling approach (e.g., concentric disk sampling).
            // We'll do a simple naive approach:
            float r = radius * sqrt(m_rng[photonID].nextFloat());
            float theta = 2.f * M_PI * m_rng[photonID].nextFloat();


            glm::vec3 refVec = sampleRandomDirection(photonID);
            // Ensure refVec is not parallel or nearly parallel to normal
            if (abs(dot(normal, refVec)) > 0.999f) {
                refVec = glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)); // Use a fixed backup vector
            }
            // Construct orthonormal basis for the disk plane
            glm::vec3 u = normalize(cross(normal, refVec));
            glm::vec3 v = cross(normal, u);

            float dx = r * cos(theta);
            float dy = r * sin(theta);

            glm::vec3 offset = dx * u + dy * v;
            return center + offset;
        }

        bool checkCameraPlaneIntersection(
            const glm::vec3 &rayOriginWorld,
            const glm::vec3 &rayDirWorld,
            glm::vec3 &hitPointCam, // out: intersection in camera space
            float &tIntersect, // out: parameter t
            float &contributionScore // out: parameter contributionScore
        ) const {
            // 1) Transform to camera space

            glm::mat4 entityTransform = m_cameraTransform->getTransform();
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

            glm::mat4 view = glm::inverse(entityTransform);
            glm::vec4 intersectionPointCamera = view * glm::vec4(intersectionPoint, 1.0f);
            glm::vec3 intersectionCamSpace = glm::vec3(intersectionPointCamera) / intersectionPointCamera.w;

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
        void accumulateOnSensor(size_t photonID, const glm::vec3 &hitPointCam, float photonFlux) const {
            // 2. Project to the image plane using pinhole intrinsics:
            // Important: Z_cam should be > 0 for a point in front of the camera.
            //

            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float X = hitPointCam.x;
            float Y = hitPointCam.y;
            float Z = hitPointCam.z;
            float xPixel = (fx * X / Z) + cx;
            float yPixel = (fy * Y / Z) + cy;
            // 2. Determine the neighboring pixels.
            // Compute the lower-left (floor) pixel coordinates and the fractional offsets.
            int x0 = static_cast<int>(std::floor(xPixel));
            int y0 = static_cast<int>(std::floor(yPixel));
            float dx = xPixel - x0;
            float dy = yPixel - y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // 3. Compute the bilinear weights for the 4 pixels.
            float w00 = (1.0f - dx) * (1.0f - dy); // weight for pixel at (x0, y0)
            float w10 = dx * (1.0f - dy); // weight for pixel at (x1, y0)
            float w01 = (1.0f - dx) * dy; // weight for pixel at (x0, y1)
            float w11 = dx * dy; // weight for pixel at (x1, y1)

            // 4. Pre-correct the photon flux with gamma adjustment.
            float correctedFlux = std::pow(photonFlux, 1.0f / m_gpuData.renderInformation->gamma);

            // 5. Retrieve image dimensions.
            const size_t imageWidth = m_camera->parameters().width;
            const size_t imageHeight = m_camera->parameters().height;

            // Helper lambda to add the weighted flux contribution to a given pixel.
            auto addFluxToPixel = [&](int px, int py, float weight) {
                // Only update if the pixel is inside the image bounds.
                if (px >= 0 && px < static_cast<int>(imageWidth) &&
                    py >= 0 && py < static_cast<int>(imageHeight)) {
                    size_t pixelIndex = static_cast<size_t>(py) * imageWidth + static_cast<size_t>(px);
                    float fluxToAdd = weight * correctedFlux;

                    // Use atomic operations to safely update the pixel value.
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            imageMemoryAtomic(m_gpuData.imageMemory[pixelIndex]);

                    // Optionally, prevent saturation by clamping the pixel value to 1.0f.
                    float currentValue = imageMemoryAtomic.load();
                    float newValue = std::min(1.0f, currentValue + fluxToAdd);
                    fluxToAdd = newValue - currentValue; // Adjust flux to the remaining margin.
                    imageMemoryAtomic.fetch_add(fluxToAdd);
                }
            };
            // 6. Distribute the corrected flux into the four neighboring pixels.
            addFluxToPixel(x0, y0, w00);
            addFluxToPixel(x1, y0, w10);
            addFluxToPixel(x0, y1, w01);
            addFluxToPixel(x1, y1, w11);
            // 7. Atomically update the photon count.
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    photonsAccumulatedAtomic(m_gpuData.renderInformation->photonsAccumulated);
            photonsAccumulatedAtomic.fetch_add(static_cast<uint64_t>(1));
        }

        // ---------------------------------------------------------------------
        //  Helper: sample an emissive gaussian object
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveGaussian(size_t photonID, size_t &entityID) const {
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
        //  sampleGaussianPositionAndNormal
        // ---------------------------------------------------------------------
        void sampleGaussianPositionAndNormal(size_t entityID, size_t emissiveEntityIdx,
                                             size_t photonID,
                                             glm::vec3 &outPos,
                                             glm::vec3 &outNormal,
                                             float &emissionPower) const {
            const GaussianInputAssembly &gaussian = m_gpuData.gaussianInputAssembly[emissiveEntityIdx];
            // ------------------------------------------------------------------
            // 1. Prepare the normal, find two tangent vectors for the plane.
            // ------------------------------------------------------------------
            glm::vec3 n = glm::normalize(gaussian.normal);
            glm::vec3 t1;
            glm::vec3 t2;
            buildTangentBasis(n, t1, t2);
            // 2. Repeatedly draw samples from a standard normal, then scale
            //    them by (sigma_x, sigma_y), until they fall inside the ellipse.

            float x, y; // final offsets in local 2D coords

            // (a) Generate two uniform randoms in [0,1)
            float u1 = m_rng[photonID].nextFloat();
            float u2 = m_rng[photonID].nextFloat();

            // (b) Box-Muller transform for standard normal
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * M_PIf * u2;
            float z0 = r * cosf(theta); // ~ N(0,1)
            float z1 = r * sinf(theta); // ~ N(0,1)

            // (c) Scale by anisotropic stddev (sigma_x, sigma_y)
            x = z0 * gaussian.scale.x;
            y = z1 * gaussian.scale.y;

            // (d) Check elliptical boundary
            //     If scale.x=1 => maximum distance is 1 meter in X
            //     If scale.y=1 => maximum distance is 1 meter in Y
            //     For ellipse: (x/σx)^2 + (y/σy)^2 <= 1
            float ellipseParam = (x * x) / (gaussian.scale.x * gaussian.scale.x)
                                 + (y * y) / (gaussian.scale.y * gaussian.scale.y);

            // ------------------------------------------------------------------
            // 3. Offset the center by (x, y) in the plane spanned by (t1, t2).
            // ------------------------------------------------------------------
            glm::vec3 offset = x * t1 + y * t2;
            outPos = gaussian.position + offset;

            // Normal remains the same as the Gaussian's normal
            outNormal = n;

            // Emission power (or flux) from your stored value
            // ------------------------------------------------------------------
            // 4. Compute Emission Power per Sample
            // ------------------------------------------------------------------
            // Total emission power from the Gaussian
            float P_total = gaussian.emission;

            // Compute the Gaussian PDF at the sampled (x, y)
            float sigma_x = gaussian.scale.x;
            float sigma_y = gaussian.scale.y;

            // Gaussian PDF (unnormalized since we are within the ellipse)
            float gaussianPDF = (1.0f / (2.0f * M_PIf * sigma_x * sigma_y)) *
                                expf(-0.5f * ((x * x) / (sigma_x * sigma_x) + (y * y) / (sigma_y * sigma_y)));

            // Area of the ellipse
            float ellipseArea = M_PIf * sigma_x * sigma_y;

            // Since we are using rejection sampling, the samples are uniformly distributed over the ellipse
            // Probability density of the uniform distribution over the ellipse
            float uniformPDF = 1.0f / ellipseArea;

            // Weight for the sample based on the ratio of Gaussian PDF to uniform PDF
            // This ensures that emissionPower reflects the importance of the sample
            float weight = gaussianPDF / uniformPDF; // = (1 / (2πσxσy)) * exp(...) / (1 / πσxσy) = 0.5 * exp(...)

            // Emission power per sample
            // Distribute P_total across samples based on weight
            emissionPower = P_total * weight;
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
        static void buildTangentBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B) {
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
            /*
                        float r = std::sqrt(u1);
                        float phi = 2.0f * M_PIf * u2; // M_PIf = float version of pi

                        // Step 2: Local coordinates (z up)
                        float x = r * std::cos(phi);
                        float y = r * std::sin(phi);
                        float z = std::sqrt(1.0f - u1);
                        */
            float theta = sycl::sqrt(u1);
            float phi = 2.0f * M_PIf * u2;
            // Step 3: Build a local orthonormal basis around 'normal'
            glm::vec3 t, b;
            buildTangentBasis(normal, t, b);

            glm::vec3 sampleWorld;
            sampleWorld.x = cos(phi) * sin(theta);
            sampleWorld.y = cos(theta);
            sampleWorld.z = sin(phi) * sin(theta);
            // Step 4: Transform from local [x, y, z] into world space
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
        glm::vec3 sampleRandomDirection(size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            return glm::normalize(r);
        }
    };
}


#endif //LightTracerKernel
