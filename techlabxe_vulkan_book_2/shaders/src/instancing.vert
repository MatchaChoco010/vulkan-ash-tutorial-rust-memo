#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inOffset;
layout(location = 3) in vec4 inColor;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec4 outColor;

out gl_PerVertex { vec4 gl_Position; };

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

layout(set = 0, binding = 1) uniform InstanceParameters {
  mat4 rotation[1600];
};

void main() {
  vec4 pos =
      rotation[gl_InstanceIndex] * vec4(inPosition, 1.0) + vec4(inOffset, 0.0);

  outWorldPosition = (ubo.model * pos).xyz;

  mat3 normalModelToWorld =
      transpose(inverse(mat3(ubo.model * rotation[gl_InstanceIndex])));
  outWorldNormal = normalModelToWorld * inNormal;

  outColor = inColor;

  gl_Position = ubo.proj * ubo.view * ubo.model * pos;
}
