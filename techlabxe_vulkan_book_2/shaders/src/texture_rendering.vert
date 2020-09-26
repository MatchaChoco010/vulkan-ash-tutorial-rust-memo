#version 450

const vec3 positions[6] =
    vec3[](vec3(-4.0, 3.0, 0.0), vec3(-4.0, -3.0, 0.0), vec3(4.0, -3.0, 0.0),
           vec3(-4.0, 3.0, 0.0), vec3(4.0, -3.0, 0.0), vec3(4.0, 3.0, 0.0));
const vec2 texcoords[6] =
    vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
           vec2(1.0, 1.0), vec2(1.0, 0.0));

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec2 outTexcoord;

out gl_PerVertex { vec4 gl_Position; };

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

void main() {
  outWorldPosition = (ubo.model * vec4(positions[gl_VertexIndex], 1.0)).xyz;

  outTexcoord = texcoords[gl_VertexIndex];

  gl_Position =
      ubo.proj * ubo.view * ubo.model * vec4(positions[gl_VertexIndex], 1.0);
}
