#version 450

const vec2 positions[3] =
    vec2[](vec2(-1.0, 3.0), vec2(-1.0, -1.0), vec2(3.0, -1.0));
const vec2 texcoords[3] =
    vec2[](vec2(0.0, 2.0), vec2(0.0, 0.0), vec2(2.0, 0.0));

layout(location = 0) out vec2 outTexcoord;

out gl_PerVertex { vec4 gl_Position; };

void main() {
  outTexcoord =
      vec2(texcoords[gl_VertexIndex].x, 1 - texcoords[gl_VertexIndex].y);

  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
