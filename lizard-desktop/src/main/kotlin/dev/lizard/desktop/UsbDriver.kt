package dev.lizard.desktop

import java.io.File
import java.util.concurrent.TimeUnit
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull

// ai: WinUSB for a phone on the cable and a USB3 Vision camera, on Windows (native/usb/lizard_usb.c, libwdi's):
// ai: libusb reaches a device there only through WinUSB, which a phone (Windows' MTP driver), the phone in accessory
// ai: mode and a USB3 Vision camera (none) lack. Receive lists such a device beside the cameras; Install USB driver
// ai: binds WinUSB to it, for a phone also staged for the accessory it switches into (Google's 18D1:2D00, and 2D01's
// ai: accessory interface with adb), one administrator's prompt; a camera's whole device (its control, event and
// ai: streaming interfaces, which Aravis claims). A phone's files are then not shown by Explorer on this PC until
// ai: Remove USB driver (Settings) gives every device its own driver back. Calls block: off the UI thread.
object UsbDriver {
    // ai: a device libusb cannot open: its ids (a phone's mi the MTP interface of a composite device, -1 a whole one;
    // ai: a camera always whole), its name, which kind
    data class Device(val vid: Int, val pid: Int, val mi: Int, val name: String, val camera: Boolean) {
        val id get() = "usbdriver:%04x:%04x:%d".format(vid, pid, mi)
        val spec get() = if (mi >= 0 && !camera) "%04x:%04x:%02x".format(vid, pid, mi) else "%04x:%04x".format(vid, pid)
    }
    data class State(val devices: List<Device>, val installed: Boolean)

    // ai: the accessory's ids, staged with every phone's: 2D00 the accessory alone, 2D01 with adb (its interface 0)
    private val ACCESSORY = listOf("18d1:2d00", "18d1:2d01:00")
    private val helper: File? get() = if (Native.windows) runCatching { File(Native.nativeDir, "lizard_usb.exe") }.getOrNull()?.takeIf { it.isFile } else null
    val here get() = helper != null

    private fun run(vararg args: String): JsonObject? {
        val h = helper ?: return null
        return runCatching {
            val p = ProcessBuilder(listOf(h.path) + args).redirectErrorStream(false).start()
            val out = p.inputStream.bufferedReader().readText()
            p.waitFor(10, TimeUnit.MINUTES)
            Json.parseToJsonElement(out.trim().lines().lastOrNull { it.startsWith("{") } ?: "{}") as? JsonObject
        }.onFailure { System.err.println("receive: lizard_usb ${args.firstOrNull()}: ${it.message}") }.getOrNull()
    }

    // ai: the phones and USB3 Vision cameras with no driver libusb can use, and whether a LIZARD driver is installed;
    // ai: null off Windows or without the helper
    fun state(): State? {
        val j = run("list") ?: return null
        val devs = (j["devices"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }
        fun JsonObject.int(k: String) = (this[k] as? JsonPrimitive)?.intOrNull ?: -1
        fun JsonObject.flag(k: String) = (this[k] as? JsonPrimitive)?.booleanOrNull == true
        fun JsonObject.str(k: String) = (this[k] as? JsonPrimitive)?.contentOrNull.orEmpty()
        // ai: a phone is reached through any interface with WinUSB (adb's has it by itself with USB debugging on); the
        // ai: accessory only through its own, interface 0
        val usable = devs.filter { it.flag("winusb") }.map { it.int("vid") to it.int("pid") }.toSet()
        fun JsonObject.accessory() = int("vid") == 0x18D1 && int("pid") in 0x2D00..0x2D05
        val phones = devs.filter { it.flag("phone") && !it.flag("winusb") && if (it.accessory()) it.int("mi") <= 0 else (it.int("vid") to it.int("pid")) !in usable }
            .map { Device(it.int("vid"), it.int("pid"), it.int("mi"), it.str("desc").ifEmpty { "Phone" }, false) }
        // ai: a camera needs WinUSB on every interface Aravis claims: one of its USB3 Vision interfaces without it
        val cameras = devs.filter { it.flag("camera") && !it.flag("winusb") }
            .map { Device(it.int("vid"), it.int("pid"), -1, it.str("desc").ifEmpty { "USB3 Vision camera" }, true) }
        val installed = (j["installed"] as? JsonArray)?.isNotEmpty() == true
        return State((phones + cameras).distinctBy { it.vid to it.pid }, installed)
    }

    // ai: WinUSB on the device (a phone's also staged for its accessory): "" done, else why. The staged ids first: an
    // ai: install waits for Windows' pending device installs, which binding a plugged-in device starts (two minutes
    // ai: in the VM with the phone bound first)
    fun install(d: Device): String {
        val j = run(*(listOf("bind") + (if (d.camera) emptyList() else ACCESSORY) + d.spec).toTypedArray()) ?: return "the driver helper did not run"
        if ((j["ok"] as? JsonPrimitive)?.booleanOrNull == true) return ""
        val err = (j["error"] as? JsonPrimitive)?.contentOrNull
            ?: (j["done"] as? JsonArray)?.mapNotNull { (it as? JsonObject)?.get("error") as? JsonPrimitive }?.firstOrNull()?.contentOrNull
        return err ?: "the driver was not installed"
    }

    // ai: every LIZARD driver removed, each device given its own back: "" done, else why
    fun remove(): String {
        val j = run("unbind") ?: return "the driver helper did not run"
        if ((j["ok"] as? JsonPrimitive)?.booleanOrNull == true) return ""
        return (j["error"] as? JsonPrimitive)?.contentOrNull ?: "a driver was not removed"
    }
}
