/* Wirehair for the browser: a narrow C ABI shaped around what a Decimen frame header
 * already carries.
 *
 * The header is 22 bytes and gives us totalLen, blockLen and seq. Wirehair's V2 decoder
 * wants a 32-byte serialized wire profile instead, and every field of that profile is
 * either a pinned constant or derivable from the header EXCEPT `seed_attempt`, which the
 * encoder picks. So the sender reads its chosen seed back out of the profile and puts it in
 * the header's `sessionId` slot (a u16, previously "random per sender start", vestigial
 * under Wirehair), and the receiver rebuilds the profile from parts. Net wire format
 * change: none.
 */
#include <wirehair/wirehair.h>

#include <cstdint>
#include <cstring>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
/* The native build (liblizard/core, lizard_xfer) calls these as plain C functions. */
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

/* Pinned deliberately. If a wirehair upgrade moved WIREHAIR_V2_PROFILE_CURRENT, a sender
 * and receiver on different builds would silently disagree about the equations, and a
 * one-way optical link has no way to notice. Bumping this is a wire format break. */
constexpr uint64_t kProfileId = WIREHAIR_V2_PROFILE_CERTIFIED_2026_07;

bool fill_profile(WirehairV2Profile& p, uint64_t message_bytes, uint32_t block_bytes,
                  uint8_t seed_attempt)
{
	memset(&p, 0, sizeof p);
	p.struct_bytes = static_cast<uint32_t>(sizeof(WirehairV2Profile));
	p.profile_version = WIREHAIR_V2_PROFILE_VERSION;
	p.profile_id = kProfileId;
	p.message_bytes = message_bytes;
	p.block_bytes = block_bytes;
	p.seed_attempt = seed_attempt;
	return true;
}

/* The JS side holds codecs as opaque numbers, so this ABI speaks void*. WirehairV2Codec
 * points at an incomplete type, so the conversion has to be spelled out. */
inline WirehairV2Codec as_codec(void* p) { return static_cast<WirehairV2Codec>(p); }

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int lizard_wh_init(void)
{
	return wirehair_init() == Wirehair_Success ? 0 : -1;
}

EMSCRIPTEN_KEEPALIVE
uint32_t lizard_wh_profile_bytes(void) { return WIREHAIR_V2_PROFILE_SERIALIZED_BYTES; }

EMSCRIPTEN_KEEPALIVE
uint32_t lizard_wh_min_blocks(void) { return 2; }

EMSCRIPTEN_KEEPALIVE
uint32_t lizard_wh_max_blocks(void) { return 64000; }

/* Create an encoder over `message`. Returns the chosen seed_attempt (0..255) on success,
 * or a negative WirehairV2Result on failure. The caller must put the returned seed in the
 * frame header so the receiver can rebuild the profile. */
EMSCRIPTEN_KEEPALIVE
int lizard_wh_encoder_create(const void* message, uint32_t message_bytes, uint32_t block_bytes,
                          void** codec_out)
{
	uint8_t profile[WIREHAIR_V2_PROFILE_SERIALIZED_BYTES];
	uint32_t profile_bytes = 0;
	WirehairV2Codec codec = nullptr;

	WirehairV2Result r = wirehair_v2_encoder_create(
		message, message_bytes, block_bytes, profile, sizeof profile, &profile_bytes, &codec);
	if (r != WirehairV2_Success) return -static_cast<int>(r);

	WirehairV2Profile parsed;
	r = wirehair_v2_profile_deserialize(profile, profile_bytes, &parsed);
	if (r != WirehairV2_Success) {
		wirehair_v2_free(codec);
		return -static_cast<int>(r);
	}
	/* If the encoder ever picks a profile we did not pin, the receiver would rebuild the
	 * wrong equations. Fail loudly here rather than corrupt a transfer. */
	if (parsed.profile_id != kProfileId) {
		wirehair_v2_free(codec);
		return -static_cast<int>(WirehairV2_UnsupportedProfile);
	}

	*codec_out = codec;
	return static_cast<int>(parsed.seed_attempt);
}

/* Build a decoder from the parts a frame header carries. */
EMSCRIPTEN_KEEPALIVE
int lizard_wh_decoder_create(uint32_t message_bytes, uint32_t block_bytes, uint32_t seed_attempt,
                          void** codec_out)
{
	if (seed_attempt > 255) return -static_cast<int>(WirehairV2_BadSeed);

	WirehairV2Profile p;
	fill_profile(p, message_bytes, block_bytes, static_cast<uint8_t>(seed_attempt));

	uint8_t profile[WIREHAIR_V2_PROFILE_SERIALIZED_BYTES];
	uint32_t written = 0;
	WirehairV2Result r = wirehair_v2_profile_serialize(&p, profile, sizeof profile, &written);
	if (r != WirehairV2_Success) return -static_cast<int>(r);

	WirehairV2Codec codec = nullptr;
	r = wirehair_v2_decoder_create(profile, written, &codec);
	if (r != WirehairV2_Success) return -static_cast<int>(r);

	*codec_out = codec;
	return 0;
}

/* block_id 0..N-1 returns the source block verbatim; N.. returns repair blocks, unbounded.
 * Returns bytes written, or a negative WirehairV2Result. */
EMSCRIPTEN_KEEPALIVE
int lizard_wh_encode(void* codec, uint32_t block_id, void* out, uint32_t capacity)
{
	uint32_t written = 0;
	WirehairV2Result r = wirehair_v2_encode(as_codec(codec), block_id, out, capacity, &written);
	if (r != WirehairV2_Success) return -static_cast<int>(r);
	return static_cast<int>(written);
}

/* Returns 1 when the payload can be recovered, 0 when more blocks are needed, negative on
 * error. Duplicate block ids are idempotent. */
EMSCRIPTEN_KEEPALIVE
int lizard_wh_decode(void* codec, uint32_t block_id, const void* data, uint32_t bytes)
{
	WirehairV2Result r = wirehair_v2_decode(as_codec(codec), block_id, data, bytes);
	if (r == WirehairV2_Success) return 1;
	if (r == WirehairV2_NeedMore) return 0;
	return -static_cast<int>(r);
}

/* Only valid once lizard_wh_decode returned 1. Does NOT authenticate: the caller checks the
 * header's payloadFnv, which Decimen already does. Returns bytes written or negative. */
EMSCRIPTEN_KEEPALIVE
int lizard_wh_recover(void* codec, void* out, uint32_t capacity)
{
	uint64_t written = 0;
	WirehairV2Result r = wirehair_v2_recover(as_codec(codec), out, capacity, &written);
	if (r != WirehairV2_Success) return -static_cast<int>(r);
	return static_cast<int>(written);
}

EMSCRIPTEN_KEEPALIVE
void lizard_wh_free(void* codec) { wirehair_v2_free(as_codec(codec)); }

} // extern "C"
