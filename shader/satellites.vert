#version 450

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec4 p3d_Color;

uniform float point_size;
uniform uint selected_id;

out vec4 v_color;
flat out uint vertex_id;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_color = p3d_Color;
    vertex_id = gl_VertexID;
    if (selected_id == vertex_id) {
        gl_PointSize = max(2.0, point_size / gl_Position.w) * 5.0;
    } else {
        gl_PointSize = max(2.0, point_size / gl_Position.w);
    }
}