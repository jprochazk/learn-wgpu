use anyhow::Result;
use std::env;
use std::path::Path;

fn main() -> Result<()> {
  // copy resources into `target/{profile}/res`
  // so that resources are always relative to the binary
  println!("cargo:rerun-if-changed=res/*");

  let to = Path::new(&env::var("CARGO_MANIFEST_DIR")?)
    .join("target")
    .join(env::var("PROFILE")?)
    .join("res");

  std::fs::create_dir_all(&to)?;
  let paths = glob::glob(concat!(env!("CARGO_MANIFEST_DIR"), "/res/**/*"))?;
  for path in paths {
    let path = path?;
    std::fs::copy(&path, to.join(path.file_name().unwrap()))?;
  }

  Ok(())
}
