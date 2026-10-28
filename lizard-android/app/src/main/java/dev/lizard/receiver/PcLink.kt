package dev.lizard.receiver

import android.app.Activity
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbAccessory
import android.hardware.usb.UsbManager
import android.os.Build
import android.provider.Settings
import android.util.Log
import java.net.Inet4Address
import java.net.NetworkInterface

// ai: The PC link's state for Receive (2026-10-27, the PC decoder, liblizard/core/rx/stream.h): the phone's USB (the
// ai: USB_STATE broadcast: a cable in, and whether its tethering function, rndis or ncm, is on), the addresses a PC on
// ai: the cable reaches this phone at (a USB tether's), and Android's tethering page: no app may switch the tether
// ai: on itself (the system's TetheringManager is for privileged apps). And the USB accessory (2026-10-27): the PC's
// ai: (lizard-desktop/native/camera/aoa.cpp) opened and handed to the core's bridge
// ai: (liblizard/core/rx/stream.h streamAccessory), which carries the stream as a TCP connection does.
object PcLink {
    // ai: the strings the PC switches a phone into accessory mode with (aoa.cpp), which res/xml/accessory_filter.xml
    // ai: matches: Android starts this activity for such an accessory with its permission given
    private const val MANUFACTURER = "LIZARD"
    private const val MODEL = "PC receiver"
    private const val PERMISSION = "dev.lizard.receiver.USB_ACCESSORY"
    private var asked = false

    // ai: the PC's accessory to the bridge where none is open (MainActivity: its intent, each resume). Without the
    // ai: permission it is asked once a process; the answer comes back as PERMISSION, which tries again.
    fun openAccessory(ctx: Context, given: UsbAccessory? = null) {
        if (Native.accessoryOpen()) return
        val um = ctx.getSystemService(UsbManager::class.java) ?: return
        val acc = (listOfNotNull(given) + um.accessoryList.orEmpty()).firstOrNull { it.manufacturer == MANUFACTURER && it.model == MODEL } ?: return
        if (!um.hasPermission(acc)) {
            if (asked) return
            asked = true
            val app = ctx.applicationContext
            val back = object : BroadcastReceiver() {
                override fun onReceive(c: Context, i: Intent) { app.unregisterReceiver(this); openAccessory(app) }
            }
            if (Build.VERSION.SDK_INT >= 33) app.registerReceiver(back, IntentFilter(PERMISSION), Context.RECEIVER_NOT_EXPORTED)
            else app.registerReceiver(back, IntentFilter(PERMISSION))
            um.requestPermission(acc, PendingIntent.getBroadcast(app, 0, Intent(PERMISSION).setPackage(app.packageName), PendingIntent.FLAG_MUTABLE))
            return
        }
        val pfd = runCatching { um.openAccessory(acc) }.onFailure { Log.w(Engine.TAG, "accessory: ${it.message}") }.getOrNull() ?: return
        Log.i(Engine.TAG, "accessory: ${acc.manufacturer} ${acc.model} ${acc.version} open")
        Native.streamAccessory(pfd.detachFd())
    }

    data class Usb(val connected: Boolean, val tether: Boolean)

    const val USB_STATE = "android.hardware.usb.action.USB_STATE"

    // ai: a USB_STATE broadcast's reading: a cable in to a host, and its tethering function, rndis or ncm, on
    fun usbOf(i: Intent?) = Usb(i?.getBooleanExtra("connected", false) == true,
        i?.getBooleanExtra("rndis", false) == true || i?.getBooleanExtra("ncm", false) == true)

    // ai: the sticky broadcast's last reading
    fun usb(ctx: Context): Usb = usbOf(runCatching {
        val f = IntentFilter(USB_STATE)
        if (Build.VERSION.SDK_INT >= 33) ctx.registerReceiver(null, f, Context.RECEIVER_EXPORTED) else ctx.registerReceiver(null, f)
    }.getOrNull())

    // ai: the IPv4 addresses a PC on the cable reaches this phone at: a USB tether's (rndis, ncm and usb interfaces)
    fun usbAddresses(): List<String> = runCatching {
        NetworkInterface.getNetworkInterfaces().toList().filter { nif ->
            nif.isUp && !nif.isLoopback && (nif.name.startsWith("rndis") || nif.name.startsWith("ncm") || nif.name.startsWith("usb"))
        }.mapNotNull { nif -> nif.inetAddresses.toList().filterIsInstance<Inet4Address>().firstOrNull()?.hostAddress }
    }.getOrDefault(emptyList())

    // ai: the tethering page where the phone has one under these names (AOSP's and most vendors'), else the network settings
    fun openTethering(a: Activity) {
        val tries = listOf(
            Intent().setClassName("com.android.settings", "com.android.settings.TetherSettings"),
            Intent().setClassName("com.android.settings", "com.android.settings.Settings\$TetherSettingsActivity"),
            Intent(Settings.ACTION_WIRELESS_SETTINGS))
        for (t in tries) if (runCatching { a.startActivity(t) }.isSuccess) return
    }
}
