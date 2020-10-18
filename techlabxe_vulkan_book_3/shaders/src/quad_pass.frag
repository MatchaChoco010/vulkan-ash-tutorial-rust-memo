#version 450

layout(location = 0) in vec2 inUv;

layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() { outColor = texture(tex, inUv).rgb; }
