use crate::texture::Texture;
use anyhow::{Context, Result};
use cgmath::{InnerSpace, Vector2, Vector3};
use std::{ops::Range, path::Path};
use tobj::*;
use wgpu::util::DeviceExt;

fn v3(v: impl Into<Vector3<f32>>) -> Vector3<f32> {
  v.into()
}

fn v2(v: impl Into<Vector2<f32>>) -> Vector2<f32> {
  v.into()
}

pub trait Vertex {
  fn descriptor<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
  position: [f32; 3],
  tex_coords: [f32; 2],
  normal: [f32; 3],
  tangent: [f32; 3],
  bitangent: [f32; 3],
}

impl Vertex for ModelVertex {
  fn descriptor<'a>() -> wgpu::VertexBufferLayout<'a> {
    use std::mem;
    wgpu::VertexBufferLayout {
      array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Vertex,
      attributes: &[
        // position
        wgpu::VertexAttribute {
          offset: 0,
          shader_location: 0,
          format: wgpu::VertexFormat::Float32x3,
        },
        // uv
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
          shader_location: 1,
          format: wgpu::VertexFormat::Float32x2,
        },
        // normal
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
          shader_location: 2,
          format: wgpu::VertexFormat::Float32x3,
        },
        // tangent
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
          shader_location: 3,
          format: wgpu::VertexFormat::Float32x3,
        },
        // bitangent
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
          shader_location: 4,
          format: wgpu::VertexFormat::Float32x3,
        },
      ],
    }
  }
}

pub struct Material {
  pub name: String,
  pub diffuse_texture: Texture,
  pub normal_texture: Texture,
  pub bind_group: wgpu::BindGroup,
}

pub struct Mesh {
  pub name: String,
  pub vertex_buffer: wgpu::Buffer,
  pub index_buffer: wgpu::Buffer,
  pub num_elements: u32,
  pub material: usize,
}

pub struct Model {
  pub meshes: Vec<Mesh>,
  pub materials: Vec<Material>,
}

impl Model {
  pub fn load<P: AsRef<Path>>(
    path: P,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
  ) -> Result<Self> {
    let path = path.as_ref();
    let (obj_models, obj_materials) = tobj::load_obj(
      path,
      &LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
      },
    )?;

    let obj_materials = obj_materials?;

    // We're assuming that the texture files are stored with the obj file
    let containing_folder = path.parent().context("Directory has no parent")?;

    let mut materials = Vec::with_capacity(obj_materials.len());
    for mat in obj_materials {
      let diffuse_texture = Texture::load(
        containing_folder.join(mat.diffuse_texture),
        device,
        queue,
        false,
      )?;
      let normal_texture = Texture::load(
        containing_folder.join(mat.normal_texture),
        device,
        queue,
        true,
      )?;
      materials.push(Material {
        name: mat.name,
        bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
          layout,
          entries: &[
            // diffuse texture
            wgpu::BindGroupEntry {
              binding: 0,
              resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
            },
            wgpu::BindGroupEntry {
              binding: 1,
              resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
            },
            // normal map
            wgpu::BindGroupEntry {
              binding: 2,
              resource: wgpu::BindingResource::TextureView(&normal_texture.view),
            },
            wgpu::BindGroupEntry {
              binding: 3,
              resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
            },
          ],
          label: None,
        }),
        diffuse_texture,
        normal_texture,
      });
    }

    let mut meshes = Vec::with_capacity(obj_models.len());
    for model in obj_models {
      let num_vertices = model.mesh.positions.len() / 3;
      let mut vertices = Vec::with_capacity(num_vertices);
      for i in 0..num_vertices {
        vertices.push(ModelVertex {
          position: [
            model.mesh.positions[i * 3],
            model.mesh.positions[i * 3 + 1],
            model.mesh.positions[i * 3 + 2],
          ],
          tex_coords: [model.mesh.texcoords[i * 2], model.mesh.texcoords[i * 2 + 1]],
          normal: [
            model.mesh.normals[i * 3],
            model.mesh.normals[i * 3 + 1],
            model.mesh.normals[i * 3 + 2],
          ],
          tangent: [0.0; 3],
          bitangent: [0.0; 3],
        });
      }

      let indices = &model.mesh.indices;
      let mut triangles_included = (0..vertices.len()).collect::<Vec<_>>();

      for idx in indices.chunks(3) {
        let v = (
          vertices[idx[0] as usize],
          vertices[idx[1] as usize],
          vertices[idx[2] as usize],
        );
        let pos = (v3(v.0.position), v3(v.1.position), v3(v.2.position));
        let uv = (v2(v.0.tex_coords), v2(v.1.tex_coords), v2(v.2.tex_coords));
        let dpos = (pos.1 - pos.0, pos.2 - pos.0);
        let duv = (uv.1 - uv.0, uv.2 - uv.0);
        let r = 1.0 / (duv.0.x * duv.1.y - duv.0.y * duv.1.x);
        let tangent = (dpos.0 * duv.1.y - dpos.1 * duv.0.y) * r;
        let bitangent = (dpos.1 * duv.0.x - dpos.0 * duv.1.x) * r;

        vertices[idx[0] as usize].tangent = (tangent + v3(v.0.tangent)).into();
        vertices[idx[1] as usize].tangent = (tangent + v3(v.1.tangent)).into();
        vertices[idx[2] as usize].tangent = (tangent + v3(v.2.tangent)).into();
        vertices[idx[0] as usize].bitangent = (bitangent + v3(v.0.bitangent)).into();
        vertices[idx[1] as usize].bitangent = (bitangent + v3(v.1.bitangent)).into();
        vertices[idx[2] as usize].bitangent = (bitangent + v3(v.2.bitangent)).into();

        triangles_included[idx[0] as usize] += 1;
        triangles_included[idx[1] as usize] += 1;
        triangles_included[idx[2] as usize] += 1;
      }

      for (i, n) in triangles_included.into_iter().enumerate() {
        let denom = 1.0 / n as f32;
        let mut v = &mut vertices[i];
        v.tangent = (v3(v.tangent) * denom).normalize().into();
        v.bitangent = (v3(v.bitangent) * denom).normalize().into();
      }

      let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Vertex Buffer", path)),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
      });
      let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{:?} Index Buffer", path)),
        contents: bytemuck::cast_slice(&model.mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
      });

      meshes.push(Mesh {
        name: model.name,
        vertex_buffer,
        index_buffer,
        num_elements: model.mesh.indices.len() as u32,
        material: model.mesh.material_id.unwrap_or(0),
      });
    }

    Ok(Self { meshes, materials })
  }
}

pub trait DrawModel<'a> {
  fn draw_mesh(
    &mut self,
    mesh: &'a Mesh,
    material: &'a Material,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
  ) {
    self.draw_mesh_instanced(mesh, material, camera_bind_group, light_bind_group, 0..1)
  }
  fn draw_mesh_instanced(
    &mut self,
    mesh: &'a Mesh,
    material: &'a Material,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
    instances: Range<u32>,
  );
  fn draw_model(
    &mut self,
    model: &'a Model,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
  ) {
    self.draw_model_instanced(model, camera_bind_group, light_bind_group, 0..1)
  }
  fn draw_model_instanced(
    &mut self,
    model: &'a Model,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
    instances: Range<u32>,
  );
}
impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
  'b: 'a,
{
  fn draw_mesh_instanced(
    &mut self,
    mesh: &'b Mesh,
    material: &'a Material,
    camera_bind_group: &'b wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
    instances: Range<u32>,
  ) {
    self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    self.set_bind_group(0, &material.bind_group, &[]);
    self.set_bind_group(1, camera_bind_group, &[]);
    self.set_bind_group(2, light_bind_group, &[]);
    self.draw_indexed(0..mesh.num_elements, 0, instances);
  }

  fn draw_model_instanced(
    &mut self,
    model: &'a Model,
    camera_bind_group: &'a wgpu::BindGroup,
    light_bind_group: &'a wgpu::BindGroup,
    instances: Range<u32>,
  ) {
    for mesh in model.meshes.iter() {
      self.draw_mesh_instanced(
        mesh,
        &model.materials[mesh.material],
        camera_bind_group,
        light_bind_group,
        instances.clone(),
      );
    }
  }
}
