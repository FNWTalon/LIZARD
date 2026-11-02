// ai: The part of Aravis 0.8's C API (and GLib's, GObject's) the Aravis backend calls (2026-10-25, ported 2026-11-01),
// ai: declared here from the 0.8.30 headers (arvsystem.h, arvmisc.h, arvcamera.h, arvstream.h, arvbuffer.h, gerror.h,
// ai: gobject.h, gsignal.h) so that lizard_cam builds on every system with no Aravis or GLib headers, and loaded at run
// ai: time (aravis.cpp aravisApi). Where it is looked for, in order, the first that loads, has every call below and says
// ai: 0.8 taken (a file missing a call or of another version is closed and the next tried):
// ai:   LIZ_ARAVIS alone where it is set: the library's path, or a folder holding it; set and empty, Aravis is off;
// ai:   the folder of the module holding this code (the app's native library, cam_check), where the apps' packages put
// ai:   it: libaravis-0.8.so.0 on Linux, libaravis-0.8.0.dylib on macOS, libaravis-0.8-0.dll on Windows (its own
// ai:   folder searched for the DLLs it needs, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
// ai:   the system's: Linux the bare name (the distro's libaravis-0.8-0); macOS Homebrew's (/opt/homebrew/lib,
// ai:   /opt/homebrew/opt/aravis/lib, /usr/local/lib, /usr/local/opt/aravis/lib), then the bare name; Windows MSYS2's
// ai:   (C:\msys64\ucrt64\bin, C:\msys64\clang64\bin), then the bare name.
// ai: On Windows the g_* calls come from the libglib-2.0-0.dll and libgobject-2.0-0.dll Aravis loaded (GetProcAddress
// ai: searches one module); elsewhere dlsym through Aravis's handle finds them. Only 0.8 is taken: the 0.9 series
// ai: renames calls this list holds (create_stream_full to stream_create). Every call here is in 0.8.8 and later.
// ai: GLib's types as the ABI has them: gboolean and gint int, guint unsigned, gulong unsigned long (32 bits on Win64,
// ai: as GLib's own), an enum an int; every object opaque.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace lizard::cam::arv {

using gboolean = int;
using gint = int;
using guint = unsigned int;
using gint64 = int64_t;
using guint64 = uint64_t;
using guint32 = uint32_t;
using gulong = unsigned long;
using PixelFormat = uint32_t;   // ai: ArvPixelFormat, a PFNC value

// ai: glib/gerror.h: domain at 0, code at 4, message at 8 on a 64-bit system
struct GError { guint32 domain; gint code; char* message; };

struct Camera;
struct Device;
struct Stream;
struct Buffer;

// ai: arvbuffer.h ArvBufferStatus
enum BufferStatus : int {
  BUFFER_UNKNOWN = -1, BUFFER_SUCCESS = 0, BUFFER_CLEARED, BUFFER_TIMEOUT, BUFFER_MISSING_PACKETS,
  BUFFER_WRONG_PACKET_ID, BUFFER_SIZE_MISMATCH, BUFFER_FILLING, BUFFER_ABORTED, BUFFER_PAYLOAD_NOT_SUPPORTED,
};
// ai: arvenums.h ArvAuto; arvgvstream.h ArvGvStreamSocketBuffer
enum Auto : int { AUTO_OFF = 0, AUTO_ONCE = 1, AUTO_CONTINUOUS = 2 };
enum : int { SOCKET_BUFFER_FIXED = 0, SOCKET_BUFFER_AUTO = 1 };

// ai: arvenums.h ARV_PIXEL_FORMAT_BIT_PER_PIXEL: a PFNC value's bits a pixel
constexpr unsigned bitsPerPixel(PixelFormat pf) { return (pf >> 16) & 0xff; }

using StreamCallback = void (*)(void* user, int type, Buffer* buffer);
using GCallback = void (*)();
using GClosureNotify = void (*)(void* data, void* closure);

// ai: every call by its C name: (return, name, parameters)
#define LIZ_ARV_CALLS(X)                                                                                              \
  X(guint, arv_get_major_version, ())                                                                                \
  X(guint, arv_get_minor_version, ())                                                                                \
  X(guint, arv_get_micro_version, ())                                                                                \
  X(void, arv_disable_interface, (const char* id))                                                                   \
  X(void, arv_update_device_list, ())                                                                                \
  X(unsigned int, arv_get_n_devices, ())                                                                             \
  X(const char*, arv_get_device_id, (unsigned int i))                                                                \
  X(const char*, arv_get_device_address, (unsigned int i))                                                           \
  X(const char*, arv_get_device_vendor, (unsigned int i))                                                            \
  X(const char*, arv_get_device_model, (unsigned int i))                                                             \
  X(const char*, arv_get_device_serial_nbr, (unsigned int i))                                                        \
  X(const char*, arv_get_device_protocol, (unsigned int i))                                                          \
  X(Camera*, arv_camera_new, (const char* name, GError** e))                                                         \
  X(Device*, arv_camera_get_device, (Camera* c))                                                                     \
  X(Stream*, arv_camera_create_stream, (Camera* c, StreamCallback cb, void* user, GError** e))                       \
  X(const char*, arv_camera_get_vendor_name, (Camera* c, GError** e))                                                \
  X(const char*, arv_camera_get_model_name, (Camera* c, GError** e))                                                 \
  X(const char*, arv_camera_get_device_serial_number, (Camera* c, GError** e))                                       \
  X(gboolean, arv_camera_is_gv_device, (Camera* c))                                                                  \
  X(gboolean, arv_camera_is_uv_device, (Camera* c))                                                                  \
  X(void, arv_camera_set_region, (Camera* c, gint x, gint y, gint w, gint h, GError** e))                           \
  X(void, arv_camera_get_region, (Camera* c, gint* x, gint* y, gint* w, gint* h, GError** e))                       \
  X(void, arv_camera_get_width_bounds, (Camera* c, gint* lo, gint* hi, GError** e))                                  \
  X(void, arv_camera_get_height_bounds, (Camera* c, gint* lo, gint* hi, GError** e))                                 \
  X(void, arv_camera_set_pixel_format, (Camera* c, PixelFormat pf, GError** e))                                      \
  X(PixelFormat, arv_camera_get_pixel_format, (Camera* c, GError** e))                                               \
  X(gint64*, arv_camera_dup_available_pixel_formats, (Camera* c, guint* n, GError** e))                              \
  X(const char**, arv_camera_dup_available_pixel_formats_as_strings, (Camera* c, guint* n, GError** e))              \
  X(const char**, arv_camera_dup_available_triggers, (Camera* c, guint* n, GError** e))                              \
  X(void, arv_camera_clear_triggers, (Camera* c, GError** e))                                                       \
  X(const char**, arv_camera_dup_available_trigger_sources, (Camera* c, guint* n, GError** e))                       \
  X(void, arv_camera_software_trigger, (Camera* c, GError** e))                                                     \
  X(void, arv_camera_start_acquisition, (Camera* c, GError** e))                                                     \
  X(void, arv_camera_stop_acquisition, (Camera* c, GError** e))                                                      \
  X(gboolean, arv_camera_is_frame_rate_available, (Camera* c, GError** e))                                           \
  X(void, arv_camera_set_frame_rate, (Camera* c, double fps, GError** e))                                            \
  X(double, arv_camera_get_frame_rate, (Camera* c, GError** e))                                                      \
  X(void, arv_camera_get_frame_rate_bounds, (Camera* c, double* lo, double* hi, GError** e))                         \
  X(gboolean, arv_camera_is_exposure_time_available, (Camera* c, GError** e))                                        \
  X(gboolean, arv_camera_is_exposure_auto_available, (Camera* c, GError** e))                                        \
  X(void, arv_camera_set_exposure_time, (Camera* c, double us, GError** e))                                          \
  X(double, arv_camera_get_exposure_time, (Camera* c, GError** e))                                                   \
  X(void, arv_camera_get_exposure_time_bounds, (Camera* c, double* lo, double* hi, GError** e))                      \
  X(void, arv_camera_set_exposure_time_auto, (Camera* c, int mode, GError** e))                                      \
  X(gboolean, arv_camera_is_gain_available, (Camera* c, GError** e))                                                 \
  X(gboolean, arv_camera_is_gain_auto_available, (Camera* c, GError** e))                                            \
  X(void, arv_camera_set_gain, (Camera* c, double gain, GError** e))                                                 \
  X(double, arv_camera_get_gain, (Camera* c, GError** e))                                                            \
  X(void, arv_camera_get_gain_bounds, (Camera* c, double* lo, double* hi, GError** e))                               \
  X(void, arv_camera_set_gain_auto, (Camera* c, int mode, GError** e))                                               \
  X(guint, arv_camera_get_payload, (Camera* c, GError** e))                                                          \
  X(void, arv_camera_set_boolean, (Camera* c, const char* feature, gboolean v, GError** e))                          \
  X(gboolean, arv_camera_get_boolean, (Camera* c, const char* feature, GError** e))                                  \
  X(void, arv_camera_set_string, (Camera* c, const char* feature, const char* v, GError** e))                        \
  X(const char*, arv_camera_get_string, (Camera* c, const char* feature, GError** e))                                \
  X(gboolean, arv_camera_is_feature_available, (Camera* c, const char* feature, GError** e))                         \
  X(guint, arv_camera_gv_auto_packet_size, (Camera* c, GError** e))                                                  \
  X(void, arv_stream_push_buffer, (Stream* s, Buffer* b))                                                            \
  X(Buffer*, arv_stream_timeout_pop_buffer, (Stream* s, guint64 timeoutUs))                                          \
  X(Buffer*, arv_buffer_new_allocate, (size_t size))                                                                 \
  X(int, arv_buffer_get_status, (Buffer* b))                                                                         \
  X(const void*, arv_buffer_get_image_data, (Buffer* b, size_t* size))                                               \
  X(gint, arv_buffer_get_image_width, (Buffer* b))                                                                   \
  X(gint, arv_buffer_get_image_height, (Buffer* b))                                                                  \
  X(void, arv_buffer_get_image_padding, (Buffer* b, gint* x, gint* y))                                               \
  X(PixelFormat, arv_buffer_get_image_pixel_format, (Buffer* b))                                                     \
  X(guint64, arv_buffer_get_frame_id, (Buffer* b))                                                                   \
  X(guint64, arv_buffer_get_timestamp, (Buffer* b))                                                                  \
  X(void, g_error_free, (GError* e))                                                                                 \
  X(void, g_free, (void* p))                                                                                         \
  X(void, g_object_unref, (void* o))                                                                                 \
  X(void, g_object_set, (void* o, const char* first, ...))                                                           \
  X(gulong, g_signal_connect_data, (void* o, const char* signal, GCallback cb, void* data, GClosureNotify destroy, int flags)) \
  X(void, g_signal_handler_disconnect, (void* o, gulong id))

struct Api {
#define LIZ_ARV_MEMBER(ret, name, args) ret(*name) args = nullptr;
  LIZ_ARV_CALLS(LIZ_ARV_MEMBER)
#undef LIZ_ARV_MEMBER
  guint vmajor = 0, vminor = 0, vmicro = 0;   // ai: arv_get_*_version's (glibc's sysmacros own major and minor)
  std::string from;   // ai: the file it was loaded from
};

}  // namespace lizard::cam::arv

namespace lizard::cam {

// ai: Aravis, loaded once a process on the first call (thread-safe): its calls, or null with why (not found, not 0.8,
// ai: a call missing; "" where LIZ_ARAVIS is set and empty: off); the same answer on every later call.
const arv::Api* aravisApi(std::string* why = nullptr);

}  // namespace lizard::cam
