#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;

out gl_PerVertex { vec4 gl_Position; };

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

void main() {
  outWorldPosition = (ubo.model * vec4(inPosition, 1.0)).xyz;

  mat3 normalModelToWorld = transpose(inverse(mat3(ubo.model)));
  outWorldNormal = normalModelToWorld * inNormal;

  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
}
