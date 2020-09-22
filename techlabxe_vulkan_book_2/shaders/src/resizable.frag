#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;

layout(location = 0) out vec4 outColor;

const vec3 LightPosition = vec3(0, 5, 5);

void main() {
  vec3 normal = normalize(inWorldNormal);
  vec3 light = normalize(LightPosition - inWorldPosition);
  float diffuse = max(dot(normal, light), 0);
  outColor = vec4(diffuse.xxx, 1.0);
}
