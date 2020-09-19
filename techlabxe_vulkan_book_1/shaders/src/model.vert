#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec2 outUV;

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 world;
  mat4 view;
  mat4 proj;
}
ubo;

out gl_PerVertex { vec4 gl_Position; };

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.world * vec4(inPos, 1.0);
  outUV = inUV;
}
