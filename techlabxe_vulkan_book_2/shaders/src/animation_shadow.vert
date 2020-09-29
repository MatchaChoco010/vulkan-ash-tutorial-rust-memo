#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in uvec2 inBlendIndices;
layout(location = 4) in vec2 inBlendWeights;
layout(location = 5) in uint inEdgeFlag;

out gl_PerVertex { vec4 gl_Position; };

layout(set = 0, binding = 0) uniform SceneParameter {
  mat4 view;
  mat4 proj;
  vec4 lightDirection;
  vec4 eyePosition;
  mat4 lightViewProj;
  mat4 lightViewProjBias;
};

layout(set = 0, binding = 1) uniform BoneParameter { mat4 boneMatrices[512]; };

vec4 TransformPosition(vec4 position) {
  vec4 pos = vec4(0);
  for (int i = 0; i < 2; ++i) {
    mat4 mtx = boneMatrices[inBlendIndices[i]];
    pos += (mtx * position) * inBlendWeights[i];
  }
  return pos;
}
vec3 TransformNormal() {
  vec3 nrm = vec3(0);
  for (int i = 0; i < 2; ++i) {
    mat4 mtx = boneMatrices[inBlendIndices[i]];
    nrm += (mat3(mtx) * inNormal) * inBlendWeights[i];
  }
  return normalize(nrm);
}

void main() {
  vec4 worldPos = TransformPosition(vec4(inPosition, 1.0));
  gl_Position = lightViewProj * worldPos;
}
