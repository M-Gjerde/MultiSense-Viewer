//
// Created by magnus on 1/28/25.
//

#ifndef PATHTRACER2DGSKERNELBACKWARD_H
#define PATHTRACER2DGSKERNELBACKWARD_H

#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"

namespace VkRender::PathTracer {
    class LightTracerKernelBackward {
    public:
        LightTracerKernelBackward(GPUData gpuData,
                                  uint32_t numPhotons,
                                  TransformComponent cameraPose,
                                  PinholeCamera* camera,
                                  uint32_t maxBounces,
                                  PCG32* rng)
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

        PCG32* m_rng;
        TransformComponent m_cameraTransform{};
        PinholeCamera* m_camera{};

        // ---------------------------------------------------------
        // Single Photon Trace (Multi-Bounce)
        // ---------------------------------------------------------
        void traceOnePhoton(size_t photonID) const {
            auto cameraTransform = m_cameraTransform.getTransform();
            glm::mat4 worldToCamera = glm::inverse(cameraTransform);

            // Camera plane normal in world space
            glm::vec3 cameraNormal = glm::normalize(
                glm::mat3(cameraTransform) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 pinholePosition = m_cameraTransform.getPosition();

            glm::vec3 cameraPlanePointWorld = glm::vec3(
    cameraTransform *
    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)); // A point on the plane

            glm::vec3 gaussianPosition = m_gpuData.gaussianInputAssembly[0].position;
            glm::vec3 gaussianNormal = m_gpuData.gaussianInputAssembly[0].normal;
            glm::vec2 gaussianScale = m_gpuData.gaussianInputAssembly[0].scale;
            glm::vec3 emissionOrigin, emissionDir;
            float emissionPower = m_gpuData.gaussianInputAssembly[0].emission;
            sampleGaussianPositionAndNormal(gaussianPosition, gaussianNormal, gaussianScale, photonID,
                                            emissionOrigin,
                                            emissionDir, emissionPower);

            float apertureDiameter = (10.0 / 4.0) / 1000;
            float apertureRadius = apertureDiameter * 0.5f;

            glm::vec3 apertureHitPoint;

            glm::vec3 emissionDirection = sampleDirectionTowardAperture(
                emissionOrigin, pinholePosition, cameraNormal,
                apertureRadius, apertureHitPoint, photonID);
            float e_tmin = FLT_MAX;
            float cosAngleHit = 0.0f;
            glm::vec3 cameraHitPointWorld(0.0f);
            checkCameraPlaneIntersection(emissionOrigin, emissionDirection,
                                       cameraHitPointWorld, e_tmin, cosAngleHit);

            // For clarity, rename e_o = emissionOrigin, e_d = apertureSampleDir
            glm::vec3 e_o = emissionOrigin;
            glm::vec3 e_d = emissionDirection;
            glm::vec3 a = apertureHitPoint;

            // Then your camera transform / projection to pixel
            // (some steps might differ if you do them in camera space first)
            glm::mat4 cameraToWorld = m_cameraTransform.getTransform();
            glm::vec4 hitCam = worldToCamera * glm::vec4(cameraHitPointWorld, 1.0f);
            glm::vec3 hitCam3 = glm::vec3(hitCam) / hitCam.w;
            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float px = hitCam3.x * 1000.f;
            float py = hitCam3.y * 1000.f;
            float pz = hitCam3.z * 1000.f;

            float xPixel = (fx * px / pz) + cx;
            float yPixel = (fy * py / pz) + cy;
            int pxInt = std::round(xPixel);
            int pyInt = std::round(yPixel);

            if (pxInt >= 0 && pxInt < (int)m_camera->parameters().width &&
                pyInt >= 0 && pyInt < (int)m_camera->parameters().height) {
                // Example: add the emission to that pixel
                // ...
                // Now do the backward pass to accumulate ∂L/∂e_o:
                // 1) Get dLoss/dI[u,v] from your gradient buffer
                int pixelIndex = pyInt * static_cast<int>(m_camera->parameters().width) + pxInt;
                float dLoss = m_gpuData.gradientImage[pixelIndex];
                // 2) We need the local partial derivative: (u,v) w.r.t. e_o
                //    We'll replicate the chain rule steps with your known transformations.
                //    The code below is just a skeleton; fill in details carefully.
                // 2.1) Emission Direction Derivative:
                // (a) r = a - e_o, e_d = r / norm(r)
                //     J_{e_d,e_o} = ...
                glm::vec3 r = a- e_o;
                float r_length = glm::length(r);
                glm::mat3x3 Jed_eo = (glm::outerProduct(r, r) / (r_length * r_length * r_length)) - (1 / r_length) * glm::mat3(1.0f);
                // 2.2) Focal Plane intersection parameter:
                // (b) tMin = ...
                //     dtMin/de_o = ...
                glm::vec3 f_n = cameraNormal;
                glm::vec3 f = cameraPlanePointWorld;
                // Dot product for denominator
                float denom = glm::dot(e_d, f_n); // Scalar
                float denom_squared = denom * denom; // Avoid recomputing later
                glm::vec3 term1 = -f_n * denom; // Scalar * Vector = Vector
                glm::vec3 term2 = (glm::dot(f, f_n) - glm::dot(e_o, f_n)) * (Jed_eo * f_n); // Scalar * (Matrix * Vector) = Vector
                glm::vec3 etmin_de_o = (term1 - term2) / denom_squared; // Element-wise division

                // 2.3) Derivatives for intersections with the focal plane
                // (c) p = e_o + tMin * e_d
                //     dp/de_o = ...
                // Identity matrix (3x3)
                glm::mat3 I(1.0f);
                // Compute first term: I
                glm::mat3 dp_de_o = I;
                // Compute second term: (∂e_tmin / ∂e_o) * e_d (3x3 * 3x1 = 3x3)
                dp_de_o += glm::outerProduct(e_d, etmin_de_o);
                // Compute third term: e_tmin * (∂e_d / ∂e_o)  (scalar * 3x3 = 3x3)
                dp_de_o += e_tmin * Jed_eo;
                // 2.4) Derivatives for the pinhole projection
                // (d) project p -> (u,v).  Then chain:
                //     d(u,v)/de_o = J_{(u,v),p} * dp/de_o
                float px_camera = hitCam3.x;
                float py_camera = hitCam3.y;
                float pz_camera = hitCam3.z;
                // Compute derivatives
                float inv_pz = 1.0f / pz_camera;
                float inv_pz2 = inv_pz * inv_pz; // 1/pz^2
                // Construct Jacobian matrix J_(u,v),p (2x3)
                glm::mat2x3 J_uv_p;
                J_uv_p[0][0] = fx * inv_pz;   // ∂u/∂px
                J_uv_p[0][1] = 0.0f;          // ∂u/∂py
                J_uv_p[0][2] = -fx * px_camera * inv_pz2; // ∂u/∂pz

                J_uv_p[1][0] = 0.0f;          // ∂v/∂px
                J_uv_p[1][1] = fy * inv_pz;   // ∂v/∂py
                J_uv_p[1][2] = -fy * py_camera * inv_pz2; // ∂v/∂pz

                // PSEUDO: Suppose we do it in world space
                // build the relevant Jacobians:

                // 5) chain them: J_uv_eo = J_uv_p * dp_de_o => (2×3) * (3×3) = 2×3
                glm::mat2x3 J_uv_eo(0.0f);
                for (int row = 0; row < 2; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        // dot across the 3 in the middle
                        float sumVal = 0.f;
                        for (int k = 0; k < 3; ++k) {
                            sumVal += J_uv_p[row][k] * dp_de_o[k][col];
                        }
                        J_uv_eo[row][col] = sumVal;
                    }
                }
                // 6) Multiply that by dLoss/dI if your pixel's intensity is
                //    "I[u,v] += emissionPower" or something similar.
                //    If you have dLoss/d(u,v) as well, you would incorporate that,
                //    but let's assume your gradientImage is effectively dL/dI.

                    // We want dL/d(e_o) = dLoss_dI * dI/d(e_o).
                    // If "I" is just the 1-pixel deposit, then dI/d(u,v) ~ 1 in your discrete sense.
                    // => dL/d(e_o) = dLoss_dI * [ ∂u/∂e_o , ∂v/∂e_o ] basically.
                    // The result is a 1×3 vector. We can combine the two rows:

                    // row0 = partial u / partial e_o
                    glm::vec3 dU_deo = { J_uv_eo[0][0], J_uv_eo[0][1], J_uv_eo[0][2] };
                    // row1 = partial v / partial e_o
                    glm::vec3 dV_deo = { J_uv_eo[1][0], J_uv_eo[1][1], J_uv_eo[1][2] };
                    float dx_u = dU_deo.x;
                    float dy_u = dU_deo.y;
                    float dz_u = dU_deo.z;

                    float dx_v = dV_deo.x;
                    float dy_v = dV_deo.y;
                    float dz_v = dV_deo.z;
                    // If the photon’s contribution to the pixel is direct =>
                    //   dI/d(u,v) = 1,
                    //   so dL/d(e_o) = dLoss_dI * [ dU_deo + dV_deo ]
                    // or you might choose to handle them separately:
                    glm::vec3 dL_deo = dU_deo * dLoss;

                    // Accumulate this into your global sum:
                // Assuming dL_deo is a glm::vec3 and m_gpuData.sumGradients is a pointer to glm::vec3
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    sum_x(m_gpuData.sumGradients->x),
                    sum_y(m_gpuData.sumGradients->y),
                    sum_z(m_gpuData.sumGradients->z);

                // Atomically accumulate into sumGradients
                sum_x.fetch_add(dL_deo.x);
                sum_y.fetch_add(dL_deo.y);
                sum_z.fetch_add(dL_deo.z);

            }
        }

        glm::mat2x3 multiplyMat2x3Mat3(const glm::mat2x3& A, const glm::mat3& B) const {
            glm::mat2x3 result;

            // Column-major multiplication: B is 3x3, A is 2x3
            for (int col = 0; col < 3; ++col) { // Iterate over columns of result
                for (int row = 0; row < 2; ++row) { // Iterate over rows of result
                    result[row][col] = A[0][col] * B[row][0] + // First row of A * first column of B
                                       A[1][col] * B[row][1] + // Second row of A * second column of B
                                       A[2][col] * B[row][2];  // Third row of A * third column of B
                }
            }

            return result;
        }
        bool geometryIntersection2DGS(
            size_t gaussianID,
            const glm::vec3& rayOrigin,
            const glm::vec3& rayDir,
            size_t& hitPointIdx,
            float& closest_t,
            glm::vec3& hitPointWorld,
            glm::vec3& hitNormalWorld
        ) const {
            bool hit = false;
            float epsilon = 1e-6f;

            for (uint32_t i = 0; i < m_gpuData.numGaussians; ++i) {
                if (i == gaussianID)
                    continue;

                const GaussianInputAssembly& gp = m_gpuData.gaussianInputAssembly[i];
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


        bool geometryIntersection(size_t lightEntityIdx, const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                                  size_t& hitEntity, float& closest_t, glm::vec3& hitPointWorld,
                                  glm::vec3& hitNormalWorld) const {
            bool hit = false;
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
                                    const glm::vec3& center,
                                    const glm::vec3& normal,
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

        glm::vec3 sampleDirectionTowardAperture(
            const glm::vec3& lightPos,
            const glm::vec3& apertureCenter,
            const glm::vec3& apertureNormal,
            float apertureRadius,
            glm::vec3& lensHitPoint,
            uint64_t photonID) const {
            // pick random point on the lens
            lensHitPoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
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
        bool accumulateOnSensor(size_t photonID, const glm::vec3& hitPointWorld, float photonFlux) const {
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

                m_gpuData.imageMemory[pixelIndex] += photonFlux;
                m_gpuData.renderInformation->photonsAccumulated++;

                /*
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> atomicImageMemory(
                    m_gpuData.imageMemory[pixelIndex]);
                atomicImageMemory.fetch_add(photonFlux);
                */
                // Atomic addition for imageMemory


                return true;
            }
            return false;
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
        //  sampleGaussianPositionAndNormal
        // ---------------------------------------------------------------------
        void sampleGaussianPositionAndNormal(const glm::vec3& position, const glm::vec3& normal, const glm::vec2& scale,
                                             size_t photonID,
                                             glm::vec3& outPos,
                                             glm::vec3& outNormal,
                                             float& emissionPower) const {
            // ------------------------------------------------------------------
            // 1. Prepare the normal, find two tangent vectors for the plane.
            // ------------------------------------------------------------------
            glm::vec3 n = glm::normalize(normal);
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
            x = z0 * scale.x;
            y = z1 * scale.y;

            // (d) Check elliptical boundary
            //     If scale.x=1 => maximum distance is 1 meter in X
            //     If scale.y=1 => maximum distance is 1 meter in Y
            //     For ellipse: (x/σx)^2 + (y/σy)^2 <= 1
            float ellipseParam = (x * x) / (scale.x * scale.x)
                + (y * y) / (scale.y * scale.y);

            // ------------------------------------------------------------------
            // 3. Offset the center by (x, y) in the plane spanned by (t1, t2).
            // ------------------------------------------------------------------
            glm::vec3 offset = x * t1 + y * t2;
            outPos = position + offset;

            // Normal remains the same as the Gaussian's normal
            outNormal = n;

            // Emission power (or flux) from your stored value
            // ------------------------------------------------------------------
            // 4. Compute Emission Power per Sample
            // ------------------------------------------------------------------
            // Total emission power from the Gaussian
            float P_total = emissionPower;

            // Compute the Gaussian PDF at the sampled (x, y)
            float sigma_x = scale.x;
            float sigma_y = scale.y;

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
        inline void buildTangentBasis(const glm::vec3& N, glm::vec3& T, glm::vec3& B) const {
            // Any vector not collinear with N will do for "temp"
            glm::vec3 temp = (fabs(N.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);

            T = glm::normalize(glm::cross(temp, N));
            B = glm::cross(N, T);
            // Now T, B, N is an orthonormal basis
        }

        glm::vec3 sampleCosineWeightedHemisphere(
            const glm::vec3& normal,
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
        glm::vec3 sampleRandomHemisphere(const glm::vec3& normal, size_t photonID) const {
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

#endif //PATHTRACER2DGSKERNELBACKWARD_H
