// ai: The Android decoder's WGSL to SPIR-V (liblizard/gen/gen.mjs compile), naga 30 as a library, because naga-cli
// ai: fixes two writer options that cost the iGPU twice the web's time (2026-09-29):
// ai:   zero_initialize_workgroup_memory: Native, not naga's Polyfill (one invocation of every workgroup storing a
// ai:     null composite over the whole of its workgroup memory, 30 KB in a classifier, then a barrier: the driver
// ai:     unrolls the store into thousands, which compiled for minutes and ran describe at 6.4 ms a frame against the
// ai:     web's 1.6). Native puts the zeroing in the variable's initializer, which Vulkan 1.3's
// ai:     shaderZeroInitializeWorkgroupMemory does (the device enables it; wg/wg.cpp).
// ai:   force_loop_bounding: false, not naga's default: no counter injected into every loop (Tint adds none for
// ai:     Vulkan; the web's WGSL has no unbounded loop).
// ai: The rest as naga-cli set them: SPIR-V 1.3, every bound check Restrict unless --unchecked (the nearest to Tint's
// ai: clamping; gen.mjs passes --unchecked for the app since 2026-09-30), f16 I/O
// ai: through f32 (a compute shader has none, and StorageInputOutput16 is a feature phones lack), WGSL's integer
// ai: division checks kept (the language defines a division by zero), no debug info.
//
//   wgsl2spv <in.wgsl> <out.spv> [--unchecked[=index,buffer,image]]
// ai: --unchecked: no bound checks (every policy Unchecked), or only the policies listed: index (arrays of a fixed
// ai: size: workgroup memory, locals, constants), buffer (arrays in storage and uniform buffers), image (textureLoad).
use naga::back::spv;
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};
use std::{env, fs, process::exit};

fn main() {
    let args: Vec<String> = env::args().collect();
    let usage = || -> ! { eprintln!("wgsl2spv <in.wgsl> <out.spv> [--unchecked[=index,buffer,image]]"); exit(2) };
    if args.len() < 3 || args.len() > 4 { usage(); }
    // ai: (index, buffer, image): which policies are Unchecked
    let mut off = (false, false, false);
    if args.len() == 4 {
        if args[3] == "--unchecked" { off = (true, true, true); }
        else if let Some(list) = args[3].strip_prefix("--unchecked=") {
            for k in list.split(',') {
                match k { "index" => off.0 = true, "buffer" => off.1 = true, "image" => off.2 = true, _ => usage() }
            }
        } else { usage(); }
    }
    let src = fs::read_to_string(&args[1]).unwrap_or_else(|e| { eprintln!("{}: {e}", args[1]); exit(1) });
    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| { eprintln!("{}", e.emit_to_string_with_path(&src, &args[1])); exit(1) });
    let caps = Capabilities::all() & spv::supported_capabilities();
    let info = Validator::new(ValidationFlags::all(), caps)
        .subgroup_stages(ShaderStages::all())
        .subgroup_operations(SubgroupOperationSet::all())
        .validate(&module)
        .unwrap_or_else(|e| { eprintln!("{}", e.emit_to_string_with_path(&src, &args[1])); exit(1) });
    let policy = |unchecked: bool| if unchecked { BoundsCheckPolicy::Unchecked } else { BoundsCheckPolicy::Restrict };
    let options = spv::Options {
        lang_version: (1, 3),
        // ai: DEBUG: names on every function, so gen.mjs finds its bitcast helpers (liz_h4 ...) by name to patch
        flags: spv::WriterFlags::DEBUG,
        bounds_check_policies: BoundsCheckPolicies { index: policy(off.0), buffer: policy(off.1), image_load: policy(off.2), binding_array: policy(off.0 && off.1 && off.2) },
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Native,
        force_loop_bounding: false,
        use_storage_input_output_16: false,
        ..spv::Options::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap_or_else(|e| { eprintln!("{}: {e}", args[1]); exit(1) });
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(&args[2], bytes).unwrap_or_else(|e| { eprintln!("{}: {e}", args[2]); exit(1) });
}
