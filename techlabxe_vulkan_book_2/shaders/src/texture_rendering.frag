#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() { outColor = texture(tex, inTexcoord); }
