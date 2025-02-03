#version 450


layout (binding = 0) uniform CameraUBO
{
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (set = 1, binding = 0) uniform Info {
    vec4 baseColor;
    float metallic;
    float roughness;
    float isDisparity;
    vec4 emissiveFactor;
} info;

layout (set = 1, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;


void main()
{
    outColor =  info.baseColor;
}
