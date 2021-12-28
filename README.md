# learn-wgpu

Result of following the tutorial over at [https://sotrh.github.io/learn-wgpu/](https://sotrh.github.io/learn-wgpu/).
I made a few changes, most notably the `build.rs` script copies `res` to `target/{profile}/res`, not to `OUT_DIR`,
which allows the path to the resources folder to be a constant.