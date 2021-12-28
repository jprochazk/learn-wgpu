use bytemuck::{Pod, Zeroable};
use cgmath::Vector3;
use std::ops::Range;

use crate::model::{Mesh, Model};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct LightUniform {
  position: [f32; 3],
  // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
  _padding: u32,
  color: [f32; 3],
}

impl LightUniform {
  pub fn new(position: [f32; 3], color: [f32; 3]) -> Self {
    Self {
      position,
      _padding: 0,
      color,
    }
  }

  pub fn position(&self) -> Vector3<f32> {
    self.position.into()
  }

  pub fn set_position(&mut self, position: impl Into<[f32; 3]>) {
    self.position = position.into();
  }
}

pub trait DrawLight<'a> {
  fn draw_light_mesh(
    &mut self,
    mesh: &'a Mesh,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
  ) {
    self.draw_light_mesh_instanced(mesh, camera_bind_group, light_bind_group, 0..1)
  }
  fn draw_light_mesh_instanced(
    &mut self,
    mesh: &'a Mesh,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
    instances: Range<u32>,
  );

  fn draw_light_model(
    &mut self,
    model: &'a Model,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
  ) {
    self.draw_light_model_instanced(model, camera_bind_group, light_bind_group, 0..1)
  }
  fn draw_light_model_instanced(
    &mut self,
    model: &'a Model,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
    instances: Range<u32>,
  );
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
  'b: 'a,
{
  fn draw_light_mesh_instanced(
    &mut self,
    mesh: &'b Mesh,
    camera_bind_group: &'b wgpu::BindGroup,
    light_bind_group: &'b wgpu::BindGroup,
    instances: Range<u32>,
  ) {
    self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    self.set_bind_group(0, camera_bind_group, &[]);
    self.set_bind_group(1, light_bind_group, &[]);
    self.draw_indexed(0..mesh.num_elements, 0, instances);
  }

  fn draw_light_model_instanced(
    &mut self,
    model: &'b Model,
    camera_bind_group: &'b wgpu::BindGroup,
    light_bind_group: &'b wgpu::BindGroup,
    instances: Range<u32>,
  ) {
    for mesh in &model.meshes {
      self.draw_light_mesh_instanced(mesh, camera_bind_group, light_bind_group, instances.clone());
    }
  }
}
