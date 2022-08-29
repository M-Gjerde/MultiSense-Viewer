#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 0) uniform Colors {
    vec4 objectColor;
    vec4 lightColor;
    vec4 lightPos;
    vec4 viewPos;
} colors;

layout(set = 0, binding = 2) uniform SELECT {
    float map;
} select ;

layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;


void main()
{

    float ambientStrength = 0.9;
    vec3 ambient = ambientStrength * colors.lightColor.xyz;
    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(colors.lightPos.xyz - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * colors.lightColor.xyz;

    // specular
    float specularStrength = 0.85;
    vec3 viewDir = normalize(colors.viewPos.xyz - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec3 specular = specularStrength * spec * colors.lightColor.xyz;

    vec3 result = (ambient + diffuse + specular) * colors.objectColor.xyz;
    vec3 tex = texture(samplerColorMap, inUV).rgb;
    outColor = vec4(tex, 1.0);

}