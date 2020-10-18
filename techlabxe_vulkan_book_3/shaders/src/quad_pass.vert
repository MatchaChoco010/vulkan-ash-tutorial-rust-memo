#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec2 outUv;

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 view;
  mat4 proj;
};

layout(push_constant) uniform PushConstants { mat4 world; };

void main() {
  gl_Position = proj * view * world * vec4(inPos, 1.0);
  outUv = inUv;
}
