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
                                  GPUDataOutput* gpuDataOutput,
                                  PCG32* rng)
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
        GPUDataOutput* m_gpuDataOutput{};

        PCG32* m_rng;
        TransformComponent* m_cameraTransform{};
        PinholeCamera* m_camera{};

        // ---------------------------------------------------------
        // Single Photon Trace (Multi-Bounce)
        // ---------------------------------------------------------
        void traceOnePhoton(size_t photonID) const {
            auto camera2World = m_cameraTransform->getTransform();
            glm::mat4 world2Camera = glm::inverse(camera2World);

            glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 pinholePosition = m_cameraTransform->getPosition();
            glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
            // A point on the plane

            // For clarity, rename e_o = emissionOrigin, e_d = apertureSampleDir
            glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
            glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;
            glm::vec3 a = m_gpuDataOutput[photonID].apertureHitPoint;
            glm::vec3 hitCam = m_gpuDataOutput[photonID].cameraHitPointLocal;
            float etmin = m_gpuDataOutput[photonID].emissionDirectionLength;
            size_t gaussianID = m_gpuDataOutput[photonID].gaussianID;

            glm::vec3 gaussianPosition = m_gpuData.gaussianInputAssembly[gaussianID].position;
            float emissionPower = m_gpuData.gaussianInputAssembly[gaussianID].emission;

            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float px = hitCam.x;
            float py = hitCam.y;
            float pz = hitCam.z;
            float xPixel = (fx * px / pz) + cx;
            float yPixel = (fy * py / pz) + cy;

            float dLoss = bilinearSample(m_gpuData.gradientImage,
                                         static_cast<int>(m_camera->parameters().width),
                                         static_cast<int>(m_camera->parameters().height),
                                         xPixel, yPixel);

            glm::vec3 f = cameraPlanePointWorld; // e.g., defined in your camera parameters
            glm::vec3 f_n = cameraNormal; // e.g., (0,0,1) if the focal plane faces +Z

            // PIXEL LOSS GROUND TRUTH
            glm::vec3 apertureHitPoint;
            glm::vec3 e_c_gt = gaussianPosition;
            glm::vec3 e_d_gt = sampleDirectionTowardAperture(
                e_c_gt,
                m_cameraTransform->getPosition(), // center of aperture
                cameraNormal,
                apertureHitPoint,
                0,
                photonID
            );
            // Check if contribution ray intersects geometry
            // Create contribution Rays and trace towards the camera
            // Trace our contribution ray
            glm::vec3 camHit;
            float tCam;
            float incidentAngle;
            bool cameraHit = checkCameraPlaneIntersection(e_c_gt, e_d_gt, camHit,
                                                          tCam, incidentAngle);
            glm::vec3 cameraHitPointWorld = e_c_gt + e_d_gt * tCam;

            // 1. Transform the hit point from world space to camera space
            glm::vec4 hitPointCam = world2Camera * glm::vec4(cameraHitPointWorld, 1.0f);
            hitPointCam = hitPointCam / hitPointCam.w;
            float px_gt = hitPointCam.x;
            float py_gt = hitPointCam.y;
            float pz_gt = hitPointCam.z;
            float gtPixelU = (fx * px_gt / pz_gt) + cx;
            float gtPixelV = (fy * py_gt / pz_gt) + cy;

            /// PART ONE ///
            // Get the Gaussian emitter parameters.
            float sigma = m_gpuData.gaussianInputAssembly[gaussianID].scale.x; // assume uniform sigma
            glm::vec3 e_c = gaussianPosition; // center of the Gaussian emitter (optimized parameter)

            // Compute the Gaussian intensity.
            glm::vec3 delta = e_c - e_o;
            float delta_norm_sq = glm::dot(delta, delta);
            float e_i = emissionPower * std::exp(-delta_norm_sq / (2.0f * sigma * sigma));
            // Compute derivative of the Gaussian intensity with respect to the sampled position e_o.
            // Note: de_i/de_o = e_i * (e_c - e_o) / sigma^2.
            glm::vec3 de_i_deo = (e_i / (sigma * sigma)) * delta;

            /// PART TWO ///
            // Example: add the emission to that pixel
            // ...
            // Now do the backward pass to accumulate ∂L/∂e_o:
            // 1) Get dLoss/dI[u,v] from your gradient buffer

            // 2) We need the local partial derivative: (u,v) w.r.t. e_o
            //    We'll replicate the chain rule steps with your known transformations.
            //    The code below is just a skeleton; fill in details carefully.
            // 2.1) Emission Direction Derivative:
            // (a) r = a - e_o, e_d = r / norm(r)
            //     J_{e_d,e_o} = ...
            glm::vec3 r = a - e_o;
            float r_length = glm::length(r);
            glm::mat3x3 Jed_eo = (glm::outerProduct(r, r) / (r_length * r_length * r_length)) - (1 / r_length) *
                glm::mat3(1.0f);
            // 2.2) Focal Plane intersection parameter:
            // (b) tMin = ...
            //     dtMin/de_o = ...
            // Dot product for denominator
            float denom = glm::dot(e_d, f_n); // Scalar
            float denom_squared = denom * denom; // Avoid recomputing later
            glm::vec3 term1 = -f_n * denom; // Scalar * Vector = Vector
            glm::vec3 term2 = (glm::dot(f, f_n) - glm::dot(e_o, f_n)) * (Jed_eo * f_n);
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
            dp_de_o += etmin * Jed_eo;
            // 2.4) Derivatives for the pinhole projection
            // (d) project p -> (u,v).  Then chain:
            //     d(u,v)/de_o = J_{(u,v),p} * dp/de_o
            float px_camera = hitCam.x;
            float py_camera = hitCam.y;
            float pz_camera = hitCam.z;
            // Compute derivatives
            float inv_pz = 1.0f / pz_camera;
            float inv_pz2 = inv_pz * inv_pz; // 1/pz^2
            // Construct Jacobian matrix J_(u,v),p (2x3)
            glm::mat3x3 J_uv_p(0.0f); // J_uv_p is in fact a 2x3 matrix but use a 3x3 for simple integration with glm
            J_uv_p[0][0] = fx * inv_pz; // ∂u/∂px
            J_uv_p[0][1] = 0.0f; // ∂u/∂py
            J_uv_p[0][2] = -fx * px_camera * inv_pz2; // ∂u/∂pz

            J_uv_p[1][0] = 0.0f; // ∂v/∂px
            J_uv_p[1][1] = fy * inv_pz; // ∂v/∂py
            J_uv_p[1][2] = -fy * py_camera * inv_pz2; // ∂v/∂pz

            // build the relevant Jacobians:
            // 5) chain them: J_uv_eo = J_uv_p * dp_de_o => (2×3) * (3×3) = 2×3
            // Apply Rotation:
            glm::mat3 dp_de_o_world = glm::mat3(world2Camera) * dp_de_o;
            //glm::mat2x3  J_uv_eo = multiply2x3_3x3(J_uv_p, dp_de_o_camera);
            glm::mat3  J_uv_eo =  glm::transpose(J_uv_p) * dp_de_o_world;
            // Now, combine the derivative contributions.
            // Previously, we computed dL/de_o = dLoss * de_i/de_o.
            // We add the extra term from the path-length differentiation:
            glm::vec3 grad_intensity = dLoss * de_i_deo;

            // 1) Evaluate the pixel mismatch:
            float du = (xPixel - gtPixelU);
            float dv = (yPixel - gtPixelV);

            // 2) dL/d(u) and dL/d(v) for L2 cost:
            float dLdu = 2.f * du;
            float dLdv = 2.f * dv;

            // For the geometry part, you need to pull back the loss derivative in image space through the Jacobian:
            glm::vec3 grad_geometry;
            grad_geometry.x = (dLdu * J_uv_eo[0][0] + dLdv * J_uv_eo[0][1]);
            grad_geometry.y = (dLdu * J_uv_eo[1][0] + dLdv * J_uv_eo[1][1]);
            grad_geometry.z = (dLdu * J_uv_eo[2][0] + dLdv * J_uv_eo[2][1]);

            //grad_geometry = grad_geometry * M_cam2world;
            float scaleFactor =  20000.0f;
            glm::vec3 grad_geometry_scaled = (grad_geometry * dLoss);
            // The total gradient is the sum:
            glm::vec3 grad_total = grad_intensity + grad_geometry_scaled;

            float d_uv_x = grad_geometry_scaled.x;
            float d_uv_y = grad_geometry_scaled.y;
            float d_uv_z = grad_geometry_scaled.z;

            float d_ei_x = grad_intensity.x;
            float d_ei_y = grad_intensity.y;
            float d_ei_z = grad_intensity.z;

            float d_total_x = grad_total.x;
            float d_total_y = grad_total.y;
            float d_total_z = grad_total.z;

            // Atomically accumulate the gradient.
            sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                sum_x(m_gpuData.sumGradients[gaussianID].x),
                sum_y(m_gpuData.sumGradients[gaussianID].y),
                sum_z(m_gpuData.sumGradients[gaussianID].z);

            sum_x.fetch_add(grad_total.x);
            sum_y.fetch_add(grad_total.y);
            sum_z.fetch_add(grad_total.z);
        }

        /*
        // Example: add the emission to that pixel
        // ...
        // Now do the backward pass to accumulate ∂L/∂e_o:
        // 1) Get dLoss/dI[u,v] from your gradient buffer

        // 2) We need the local partial derivative: (u,v) w.r.t. e_o
        //    We'll replicate the chain rule steps with your known transformations.
        //    The code below is just a skeleton; fill in details carefully.
        // 2.1) Emission Direction Derivative:
        // (a) r = a - e_o, e_d = r / norm(r)
        //     J_{e_d,e_o} = ...
        glm::vec3 r = a - e_o;
        float r_length = glm::length(r);
        glm::mat3x3 Jed_eo = (glm::outerProduct(r, r) / (r_length * r_length * r_length)) - (1 / r_length) *
            glm::mat3(1.0f);
        // 2.2) Focal Plane intersection parameter:
        // (b) tMin = ...
        //     dtMin/de_o = ...
        glm::vec3 f_n = cameraNormal;
        glm::vec3 f = cameraPlanePointWorld;
        // Dot product for denominator
        float denom = glm::dot(e_d, f_n); // Scalar
        float denom_squared = denom * denom; // Avoid recomputing later
        glm::vec3 term1 = -f_n * denom; // Scalar * Vector = Vector
        glm::vec3 term2 = (glm::dot(f, f_n) - glm::dot(e_o, f_n)) * (Jed_eo * f_n);
        // Scalar * (Matrix * Vector) = Vector
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
        dp_de_o += etmin * Jed_eo;
        // 2.4) Derivatives for the pinhole projection
        // (d) project p -> (u,v).  Then chain:
        //     d(u,v)/de_o = J_{(u,v),p} * dp/de_o
        float px_camera = hitCam.x;
        float py_camera = hitCam.y;
        float pz_camera = hitCam.z;
        // Compute derivatives
        float inv_pz = 1.0f / pz_camera;
        float inv_pz2 = inv_pz * inv_pz; // 1/pz^2
        // Construct Jacobian matrix J_(u,v),p (2x3)
        glm::mat2x3 J_uv_p;
        J_uv_p[0][0] = fx * inv_pz; // ∂u/∂px
        J_uv_p[0][1] = 0.0f; // ∂u/∂py
        J_uv_p[0][2] = -fx * px_camera * inv_pz2; // ∂u/∂pz

        J_uv_p[1][0] = 0.0f; // ∂v/∂px
        J_uv_p[1][1] = fy * inv_pz; // ∂v/∂py
        J_uv_p[1][2] = -fy * py_camera * inv_pz2; // ∂v/∂pz

        // PSEUDO: Suppose we do it in world space
        // build the relevant Jacobians:

        // 5) chain them: J_uv_eo = J_uv_p * dp_de_o => (2×3) * (3×3) = 2×3
        glm::mat2x3 J_uv_eo(0.0f);

        J_uv_eo = multiply2x3_3x3(J_uv_p, glm::transpose(dp_de_o)); // Transpose due to column-major memory layout

        // 6) Multiply that by dLoss/dI if your pixel's intensity is
        //    "I[u,v] += emissionPower" or something similar.
        //    If you have dLoss/d(u,v) as well, you would incorporate that,
        //    but let's assume your gradientImage is effectively dL/dI.

        // We want dL/d(e_o) = dLoss_dI * dI/d(e_o).
        // If "I" is just the 1-pixel deposit, then dI/d(u,v) ~ 1 in your discrete sense.
        // => dL/d(e_o) = dLoss_dI * [ ∂u/∂e_o , ∂v/∂e_o ] basically.
        // The result is a 1×3 vector. We can combine the two rows:

        // row0 = partial u / partial e_o
        glm::vec3 dU_deo = {J_uv_eo[0][0], J_uv_eo[0][1], J_uv_eo[0][2]};
        // row1 = partial v / partial e_o
        glm::vec3 dV_deo = {J_uv_eo[1][0], J_uv_eo[1][1], J_uv_eo[1][2]};
        float& dx_u = dU_deo.x;
        float& dy_u = dU_deo.y;
        float& dz_u = dU_deo.z;
        dz_u = 0;
        float dx_v = dV_deo.x;
        float dy_v = dV_deo.y;
        float dz_v = dV_deo.z;
        // If the photon’s contribution to the pixel is direct =>
        //   dI/d(u,v) = 1,
        //   so dL/d(e_o) = dLoss_dI * [ dU_deo + dV_deo ]
        // or you might choose to handle them separately:
        //glm::vec3 dL_deo = dU_deo * dLoss * emissionPower;

        */
        glm::mat2x3 multiply2x3_3x3(const glm::mat2x3& A, const glm::mat3& B) const {
            glm::mat2x3 result;
            // Manual matrix multiplication
            result[0][0] = A[0][0] * B[0][0] + A[1][0] * B[0][1] + A[2][0] * B[0][2];
            result[1][0] = A[0][0] * B[1][0] + A[1][0] * B[1][1] + A[2][0] * B[1][2];
            result[2][0] = A[0][0] * B[2][0] + A[1][0] * B[2][1] + A[2][0] * B[2][2];

            result[1][1] = A[1][0] * B[0][0] + A[1][1] * B[0][1] + A[1][2] * B[0][2];
            result[1][1] = A[1][0] * B[1][0] + A[1][1] * B[1][1] + A[1][2] * B[1][2];
            result[1][1] = A[1][0] * B[2][0] + A[1][1] * B[2][1] + A[1][2] * B[2][2];

            return result;
        }


        float bilinearSample(const float* image, int width, int height, float x, float y) const {
            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // Clamp to image boundaries
            x0 = std::clamp(x0, 0, width - 1);
            y0 = std::clamp(y0, 0, height - 1);
            x1 = std::clamp(x1, 0, width - 1);
            y1 = std::clamp(y1, 0, height - 1);

            // Compute interpolation weights
            float dx = x - x0;
            float dy = y - y0;

            // Fetch pixel values
            float I00 = image[y0 * width + x0];
            float I10 = image[y0 * width + x1];
            float I01 = image[y1 * width + x0];
            float I11 = image[y1 * width + x1];

            // Bilinear interpolation formula
            return (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I10 +
                (1 - dx) * dy * I01 + dx * dy * I11;
        }

        glm::vec3 sampleDirectionTowardAperture(
            const glm::vec3& lightPos,
            const glm::vec3& apertureCenter,
            const glm::vec3& apertureNormal,
            glm::vec3& apertureHitpoint,
            float apertureRadius,
            uint64_t photonID) const {
            // pick random point on the lens
            apertureHitpoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
            // direction from light to lens point
            glm::vec3 dir = apertureHitpoint - lightPos;
            return normalize(dir);
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

        bool checkCameraPlaneIntersection(
            const glm::vec3& rayOriginWorld,
            const glm::vec3& rayDirWorld,
            glm::vec3& hitPointCam, // out: intersection in camera space
            float& tIntersect, // out: parameter t
            float& contributionScore // out: parameter contributionScore
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
        void accumulateOnSensor(size_t photonID, const glm::vec3& hitPointCam, float photonFlux) const {
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
        void sampleGaussianPositionAndNormal(size_t entityID, size_t emissiveEntityIdx,
                                             size_t photonID,
                                             glm::vec3& outPos,
                                             glm::vec3& outNormal,
                                             float& emissionPower) const {
            const GaussianInputAssembly& gaussian = m_gpuData.gaussianInputAssembly[emissiveEntityIdx];
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
        static void buildTangentBasis(const glm::vec3& N, glm::vec3& T, glm::vec3& B) {
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
