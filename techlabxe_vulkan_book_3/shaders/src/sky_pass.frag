#version 460

layout(location = 0) in vec3 direction;

layout(location = 0) out vec3 outputColor;

layout(set = 0, binding = 1) uniform samplerCube cubemap;

void main() { outputColor = texture(cubemap, direction).rgb; }
