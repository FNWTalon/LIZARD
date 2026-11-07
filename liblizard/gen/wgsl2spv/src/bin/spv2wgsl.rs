// ai: The decoder's plain GLSL kernels as WGSL for WebGPU (liblizard/gen/gen.mjs webgpuOf, 2026-11-06): a module as
// ai: glslang wrote it, read by naga's SPIR-V front end, validated with no optional capability (WebGPU's core: no
// ai: 16-bit or 8-bit integers, no subgroups, no f16) and written as WGSL. One source, the GLSL; nothing kept by hand.
//
//   spv2wgsl <in.spv> <out.wgsl>
// ai: Exit 1 with naga's words where it cannot (a form using what WGSL lacks, or what naga's reader does not take).
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::{env, fs, process::exit};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 { eprintln!("spv2wgsl <in.spv> <out.wgsl>"); exit(2); }
    let fail = |what: &str, e: String| -> ! { eprintln!("{}: {what}: {e}", args[1]); exit(1) };
    let bytes = fs::read(&args[1]).unwrap_or_else(|e| fail("read", e.to_string()));
    let module = naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default()).unwrap_or_else(|e| fail("parse", e.to_string()));
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module).unwrap_or_else(|e| fail("validate", format!("{:?}", e.as_inner())));
    let wgsl = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()).unwrap_or_else(|e| fail("write", e.to_string()));
    fs::write(&args[2], wgsl).unwrap_or_else(|e| fail("write", e.to_string()));
}
