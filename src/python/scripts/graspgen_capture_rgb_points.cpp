#include "libobsensor/ObSensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Strip the .ply extension so the image siblings sit next to the cloud:
// /tmp/scene.ply -> /tmp/scene_color.jpg, /tmp/scene_depth.pgm
static std::string strip_extension(const std::string &path) {
    const size_t dot = path.find_last_of('.');
    const size_t slash = path.find_last_of('/');
    if(dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path;
    }
    return path.substr(0, dot);
}

static bool write_binary(const std::string &fileName, const void *data, size_t size) {
    FILE *fp = std::fopen(fileName.c_str(), "wb");
    if(!fp) {
        std::cerr << "cannot open " << fileName << " for writing" << std::endl;
        return false;
    }
    const bool ok = std::fwrite(data, 1, size, fp) == size;
    std::fclose(fp);
    return ok;
}

static uint8_t clamp_u8(int v) {
    return (uint8_t)std::min(255, std::max(0, v));
}

// BT.601 limited-range YUV -> RGB, which is what the UVC colour formats carry.
static void yuv_to_rgb(int y, int u, int v, uint8_t *out) {
    const int c = y - 16;
    const int d = u - 128;
    const int e = v - 128;
    out[0] = clamp_u8((298 * c + 409 * e + 128) >> 8);
    out[1] = clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
    out[2] = clamp_u8((298 * c + 516 * d + 128) >> 8);
}

// Decode whatever the colour sensor hands us into packed RGB888.
//
// MJPG is deliberately NOT handled here - it is already a complete JPEG file and
// the caller writes the bytes straight out, which avoids pulling in a decoder.
static bool color_frame_to_rgb(const std::shared_ptr<ob::ColorFrame> &color, std::vector<uint8_t> &rgb) {
    const int w = (int)color->width();
    const int h = (int)color->height();
    const uint8_t *src = (const uint8_t *)color->data();
    const size_t size = color->dataSize();
    const size_t pixels = (size_t)w * (size_t)h;
    rgb.assign(pixels * 3, 0);

    switch(color->format()) {
    case OB_FORMAT_RGB:
        if(size < pixels * 3) return false;
        std::memcpy(rgb.data(), src, pixels * 3);
        return true;
    case OB_FORMAT_BGR:
        if(size < pixels * 3) return false;
        for(size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = src[i * 3 + 2];
            rgb[i * 3 + 1] = src[i * 3 + 1];
            rgb[i * 3 + 2] = src[i * 3 + 0];
        }
        return true;
    case OB_FORMAT_YUYV:
    case OB_FORMAT_YUY2:
    case OB_FORMAT_UYVY: {
        if(size < pixels * 2) return false;
        const bool uyvy = color->format() == OB_FORMAT_UYVY;
        for(size_t i = 0; i + 1 < pixels; i += 2) {
            const uint8_t *p = src + i * 2;
            const int y0 = uyvy ? p[1] : p[0];
            const int u = uyvy ? p[0] : p[1];
            const int y1 = uyvy ? p[3] : p[2];
            const int v = uyvy ? p[2] : p[3];
            yuv_to_rgb(y0, u, v, &rgb[i * 3]);
            yuv_to_rgb(y1, u, v, &rgb[(i + 1) * 3]);
        }
        return true;
    }
    case OB_FORMAT_NV12:
    case OB_FORMAT_NV21: {
        if(size < pixels + pixels / 2) return false;
        const bool nv21 = color->format() == OB_FORMAT_NV21;
        const uint8_t *uvPlane = src + pixels;
        for(int row = 0; row < h; ++row) {
            for(int col = 0; col < w; ++col) {
                const int y = src[(size_t)row * w + col];
                const uint8_t *uv = uvPlane + ((size_t)(row / 2) * w) + (col & ~1);
                const int u = nv21 ? uv[1] : uv[0];
                const int v = nv21 ? uv[0] : uv[1];
                yuv_to_rgb(y, u, v, &rgb[((size_t)row * w + col) * 3]);
            }
        }
        return true;
    }
    default:
        return false;
    }
}

// Save the colour frame as a real viewable image. Returns the path written, or an
// empty string on failure.
//
// Formats are chosen so nothing extra has to be installed on the Pi (no OpenCV,
// no libpng): MJPG is passed through as .jpg, everything else becomes a binary
// PPM, which every image viewer and PIL/matplotlib reads.
static std::string save_color_image(const std::shared_ptr<ob::ColorFrame> &color, const std::string &stem) {
    if(color->format() == OB_FORMAT_MJPG || color->format() == OB_FORMAT_MJPEG) {
        const std::string path = stem + "_color.jpg";
        // dataSize() is the buffer capacity, not the JPEG length: the observed
        // frame carried 4 zero bytes past the end-of-image marker. Trim to EOI so
        // the file is a clean JPEG rather than one strict decoders may reject.
        const uint8_t *bytes = (const uint8_t *)color->data();
        size_t size = color->dataSize();
        for(size_t i = size; i >= 2; --i) {
            if(bytes[i - 2] == 0xFF && bytes[i - 1] == 0xD9) {
                size = i;
                break;
            }
        }
        return write_binary(path, bytes, size) ? path : std::string();
    }
    std::vector<uint8_t> rgb;
    if(!color_frame_to_rgb(color, rgb)) {
        std::cerr << "unsupported color format " << color->format() << "; image not saved" << std::endl;
        return std::string();
    }
    const std::string path = stem + "_color.ppm";
    FILE *fp = std::fopen(path.c_str(), "wb");
    if(!fp) {
        std::cerr << "cannot open " << path << " for writing" << std::endl;
        return std::string();
    }
    std::fprintf(fp, "P6\n%d %d\n255\n", (int)color->width(), (int)color->height());
    const bool ok = std::fwrite(rgb.data(), 1, rgb.size(), fp) == rgb.size();
    std::fclose(fp);
    return ok ? path : std::string();
}

// Save depth as a 16-bit binary PGM, preserving the RAW sensor values.
//
// Raw, not millimetres: multiply by the scale printed in `depth_scale` to get mm.
// Scaling here would either lose precision or need a float format, and the point
// cloud already carries metric data for anything that needs it.
static std::string save_depth_image(const std::shared_ptr<ob::DepthFrame> &depth, const std::string &stem) {
    const int w = (int)depth->width();
    const int h = (int)depth->height();
    const size_t pixels = (size_t)w * (size_t)h;
    if(depth->dataSize() < pixels * 2) {
        return std::string();
    }
    const std::string path = stem + "_depth.pgm";
    FILE *fp = std::fopen(path.c_str(), "wb");
    if(!fp) {
        std::cerr << "cannot open " << path << " for writing" << std::endl;
        return std::string();
    }
    std::fprintf(fp, "P5\n%d %d\n65535\n", w, h);
    // PGM is big-endian by specification; the sensor buffer is little-endian.
    const uint16_t *src = (const uint16_t *)depth->data();
    std::vector<uint8_t> be(pixels * 2);
    for(size_t i = 0; i < pixels; ++i) {
        be[i * 2 + 0] = (uint8_t)(src[i] >> 8);
        be[i * 2 + 1] = (uint8_t)(src[i] & 0xFF);
    }
    const bool ok = std::fwrite(be.data(), 1, be.size(), fp) == be.size();
    std::fclose(fp);
    return ok ? path : std::string();
}

static int save_rgb_points_to_ply(std::shared_ptr<ob::Frame> frame, const std::string &fileName) {
    int pointsSize = frame->dataSize() / sizeof(OBColorPoint);
    OBColorPoint *point = (OBColorPoint *)frame->data();
    int validPointsCount = 0;
    static const auto min_distance = 1e-6;
    for(int i = 0; i < pointsSize; i++, point++) {
        if(std::fabs(point->x) >= min_distance || std::fabs(point->y) >= min_distance || std::fabs(point->z) >= min_distance) {
            validPointsCount++;
        }
    }

    FILE *fp = std::fopen(fileName.c_str(), "wb+");
    if(!fp) {
        throw std::runtime_error("Failed to open file for writing");
    }
    std::fprintf(fp, "ply\n");
    std::fprintf(fp, "format ascii 1.0\n");
    std::fprintf(fp, "element vertex %d\n", validPointsCount);
    std::fprintf(fp, "property float x\n");
    std::fprintf(fp, "property float y\n");
    std::fprintf(fp, "property float z\n");
    std::fprintf(fp, "property uchar red\n");
    std::fprintf(fp, "property uchar green\n");
    std::fprintf(fp, "property uchar blue\n");
    std::fprintf(fp, "end_header\n");
    point = (OBColorPoint *)frame->data();
    for(int i = 0; i < pointsSize; i++, point++) {
        if(std::fabs(point->x) >= min_distance || std::fabs(point->y) >= min_distance || std::fabs(point->z) >= min_distance) {
            std::fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", point->x, point->y, point->z, (int)point->r, (int)point->g, (int)point->b);
        }
    }
    std::fflush(fp);
    std::fclose(fp);
    return validPointsCount;
}

// Same PLY layout as the RGB writer but for an uncoloured cloud. Colours are
// written as white rather than omitted so the header stays identical and existing
// readers (which expect x y z r g b) keep working.
static int save_xyz_points_to_ply(std::shared_ptr<ob::Frame> frame, const std::string &fileName) {
    int pointsSize = frame->dataSize() / sizeof(OBPoint);
    OBPoint *point = (OBPoint *)frame->data();
    int validPointsCount = 0;
    static const auto min_distance = 1e-6;
    for(int i = 0; i < pointsSize; i++, point++) {
        if(std::fabs(point->x) >= min_distance || std::fabs(point->y) >= min_distance || std::fabs(point->z) >= min_distance) {
            validPointsCount++;
        }
    }
    FILE *fp = std::fopen(fileName.c_str(), "wb+");
    if(!fp) {
        throw std::runtime_error("Failed to open file for writing");
    }
    std::fprintf(fp, "ply\nformat ascii 1.0\nelement vertex %d\n", validPointsCount);
    std::fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
    std::fprintf(fp, "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
    point = (OBPoint *)frame->data();
    for(int i = 0; i < pointsSize; i++, point++) {
        if(std::fabs(point->x) >= min_distance || std::fabs(point->y) >= min_distance || std::fabs(point->z) >= min_distance) {
            std::fprintf(fp, "%.3f %.3f %.3f 255 255 255\n", point->x, point->y, point->z);
        }
    }
    std::fflush(fp);
    std::fclose(fp);
    return validPointsCount;
}

int main(int argc, char **argv) try {
    const std::string output = argc > 1 ? argv[1] : "/tmp/graspgen_rgb_points.ply";
    const int min_valid = argc > 2 ? std::atoi(argv[2]) : 50000;
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);
    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();

    // The laser can come up disabled after a device power cycle, and every
    // depth frame reads all-zero until it is turned back on.
    try {
        auto device = pipeline.getDevice();
        if(device && device->isPropertySupported(OB_PROP_LASER_BOOL, OB_PERMISSION_READ_WRITE)) {
            if(!device->getBoolProperty(OB_PROP_LASER_BOOL)) {
                device->setBoolProperty(OB_PROP_LASER_BOOL, true);
                std::cout << "laser was off; enabled" << std::endl;
            }
        }
    }
    catch(ob::Error &e) {
        std::cerr << "laser property error: " << e.getMessage() << std::endl;
    }

    // Colour brightness trim, range -64..64, 0 = sensor default.
    //
    // Brightness, NOT exposure, deliberately. Writing OB_PROP_COLOR_EXPOSURE_INT
    // (which requires disabling auto-exposure first) destroys the depth stream on
    // this device: measured 2026-07-30, an otherwise identical run reported
    // nonzero=0/307200 for all 100 frames with the exposure write, and
    // nonzero=169884 on frame 0 with it removed. Turning auto-exposure off appears
    // to reconfigure the UVC pipeline in a way the depth path does not survive.
    //
    // Brightness is safe - 0, 32 and 64 all kept depth at ~170000 points - and it
    // is enough, because auto-exposure does eventually produce a correctly exposed
    // colour frame here (22204-byte JPEG, mean 219/255) once the colour stream has
    // been running. The earlier "auto-exposure is stuck at 157 and the image is
    // black" reading came from grabbing the very first colour frame.
    const int color_brightness = argc > 3 ? std::atoi(argv[3]) : 0;

    std::shared_ptr<ob::VideoStreamProfile> colorProfile = nullptr;
    try {
        auto colorProfiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
        if(colorProfiles && colorProfiles->count() > 0) {
            auto profile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
            colorProfile = profile->as<ob::VideoStreamProfile>();
            config->enableStream(colorProfile);
            std::cout << "color profile: " << colorProfile->width() << "x" << colorProfile->height()
                      << " fps=" << colorProfile->fps() << " format=" << colorProfile->format() << std::endl;
        }
    }
    catch(ob::Error &e) {
        std::cerr << "color profile error: " << e.getMessage() << std::endl;
    }
    if(!colorProfile) {
        std::cerr << "no color profile" << std::endl;
        return 2;
    }

    std::shared_ptr<ob::StreamProfileList> depthProfileList;
    OBAlignMode alignMode = ALIGN_DISABLE;
    try {
        depthProfileList = pipeline.getD2CDepthProfileList(colorProfile, ALIGN_D2C_HW_MODE);
        if(depthProfileList && depthProfileList->count() > 0) {
            alignMode = ALIGN_D2C_HW_MODE;
        }
        else {
            depthProfileList = pipeline.getD2CDepthProfileList(colorProfile, ALIGN_D2C_SW_MODE);
            if(depthProfileList && depthProfileList->count() > 0) {
                alignMode = ALIGN_D2C_SW_MODE;
            }
        }
    }
    catch(ob::Error &e) {
        std::cerr << "D2C profile error: " << e.getMessage() << std::endl;
        depthProfileList = nullptr;
    }
    if(!depthProfileList || depthProfileList->count() == 0) {
        depthProfileList = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
    }
    if(!depthProfileList || depthProfileList->count() == 0) {
        std::cerr << "no depth profiles" << std::endl;
        return 3;
    }

    std::shared_ptr<ob::StreamProfile> depthProfile;
    try {
        depthProfile = depthProfileList->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, colorProfile->fps());
    }
    catch(...) {
        depthProfile = depthProfileList->getProfile(OB_PROFILE_DEFAULT);
    }
    auto depthVideo = depthProfile->as<ob::VideoStreamProfile>();
    std::cout << "depth profile: " << depthVideo->width() << "x" << depthVideo->height()
              << " fps=" << depthVideo->fps() << " format=" << depthVideo->format()
              << " align=" << alignMode << std::endl;
    config->enableStream(depthProfile);
    config->setAlignMode(alignMode);

    try {
        pipeline.enableFrameSync();
    }
    catch(ob::Error &e) {
        std::cerr << "frame sync unavailable: " << e.getMessage() << std::endl;
    }

    pipeline.start(config);

    // Re-assert the laser AFTER start(). The pre-start write is not always kept:
    // observed 2026-07-30 reporting "laser was off; enabled" on three consecutive
    // runs and still delivering all-zero depth, so starting the streams can clear
    // it. Checking again here is cheap and the failure mode is total depth loss.
    // Written UNCONDITIONALLY, not gated on a readback. The readback lies: it can
    // report the laser on while depth is still all-zero, and it reads 0 again right
    // after a capture that produced a dense cloud. A minimal probe that always
    // wrote true here got 169812 depth points on frame 0, while this file's earlier
    // read-then-write version returned all-zero depth on four consecutive runs.
    try {
        auto device = pipeline.getDevice();
        if(device && device->isPropertySupported(OB_PROP_LASER_BOOL, OB_PERMISSION_WRITE)) {
            device->setBoolProperty(OB_PROP_LASER_BOOL, true);
            std::cout << "laser re-asserted after start" << std::endl;
        }
    }
    catch(ob::Error &e) {
        std::cerr << "post-start laser property error: " << e.getMessage() << std::endl;
    }

    // Must come AFTER start(): the colour sensor only accepts this once its stream
    // is running. Auto-exposure is left ON on purpose - see color_brightness.
    if(color_brightness != 0) {
        try {
            auto device = pipeline.getDevice();
            device->setIntProperty(OB_PROP_COLOR_BRIGHTNESS_INT, color_brightness);
            std::cout << "color brightness set to " << color_brightness << " (readback "
                      << device->getIntProperty(OB_PROP_COLOR_BRIGHTNESS_INT) << ")" << std::endl;
        }
        catch(ob::Error &e) {
            std::cerr << "color brightness error: " << e.getMessage() << std::endl;
        }
    }

    ob::PointCloudFilter pointCloud;
    pointCloud.setCameraParam(pipeline.getCameraParam());
    pointCloud.setCreatePointFormat(OB_FORMAT_RGB_POINT);

    int bestValid = -1;
    // Most recent colour frame, carried across iterations because depth and colour
    // never arrive in the same usable frameset (see the loop body).
    std::shared_ptr<ob::ColorFrame> lastColor;

    // NOTE ON ORDERING: depth is captured FIRST and colour is fetched afterwards
    // (see `fetch_color_after` below). Draining framesets up front looking for
    // colour starves the depth stream - a colour-first warm-up got colour on frame
    // 13 and then every subsequent depth frame read all-zero. Depth is the payload,
    // so it gets the healthy start of the session.
    for(int i = 0; i < 100; ++i) {
        auto frameset = pipeline.waitForFrames(500);
        if(frameset == nullptr || frameset->depthFrame() == nullptr) {
            std::cout << "frame " << i << ": missing " << (frameset == nullptr ? "frameset" : "depth") << std::endl;
            continue;
        }
        // Colour is OPTIONAL here, depth is required, and the two are effectively
        // MUTUALLY EXCLUSIVE on this device. Measured over 60 framesets
        // (2026-07-30): 11 carried both streams and 49 were depth-only, but every
        // both-streams frameset had ZERO valid depth pixels, while the depth-only
        // ones were dense (170172). The device also reports "does not support frame
        // sync". So a frameset either has usable depth or has colour, never both.
        //
        // This loop used to require both and `continue` otherwise, which burned the
        // whole 100-frame budget and reported "failed to capture" while depth was
        // dense the entire time.
        //
        // Consequence: the colour image is saved from the most recent colour frame
        // seen, which is a DIFFERENT instant from the depth frame. Fine for looking
        // at the scene; do not treat the pair as pixel-registered.
        auto colorFrame = frameset->colorFrame();
        if(colorFrame != nullptr) {
            lastColor = colorFrame;
        }
        auto depth = frameset->depthFrame();
        const uint16_t *data = (const uint16_t *)depth->data();
        int pixels = depth->width() * depth->height();
        int nonzero = 0;
        uint16_t minv = 65535;
        uint16_t maxv = 0;
        for(int j = 0; j < pixels; ++j) {
            uint16_t z = data[j];
            if(z != 0) {
                nonzero++;
                if(z < minv) minv = z;
                if(z > maxv) maxv = z;
            }
        }
        std::cout << "frame " << i << ": depth " << depth->width() << "x" << depth->height()
                  << " scale=" << depth->getValueScale()
                  << " nonzero=" << nonzero << "/" << pixels
                  << " raw_min=" << (nonzero ? minv : 0) << " raw_max=" << maxv << std::endl;
        if(nonzero < 1000) {
            continue;
        }
        try {
            pointCloud.setPositionDataScaled(depth->getValueScale());
            int valid = 0;
            if(colorFrame != nullptr) {
                pointCloud.setCreatePointFormat(OB_FORMAT_RGB_POINT);
                valid = save_rgb_points_to_ply(pointCloud.process(frameset), output);
            }
            else {
                // `PointCloudFilter` errors with "no color frame found in frameset"
                // when asked for RGB_POINT without colour, and the dense-depth
                // framesets on this device never carry colour. Geometry is what the
                // grasp pipeline consumes, so fall back to XYZ and keep the colour
                // picture as a separate file.
                pointCloud.setCreatePointFormat(OB_FORMAT_POINT);
                valid = save_xyz_points_to_ply(pointCloud.process(frameset), output);
            }
            bestValid = valid;
            std::cout << "saved " << output << " valid_points=" << valid
                      << (colorFrame != nullptr ? " (with RGB)" : " (XYZ only, no colour in this frameset)")
                      << std::endl;

            // Also keep the plain 2D images. The colour frame was already being
            // pulled in to tint the cloud, so this costs one extra write and gives
            // a picture that can actually be looked at when the geometry is
            // confusing - e.g. deciding whether a cluster is one object or several.
            const std::string stem = strip_extension(output);
            const std::string depthPath = save_depth_image(depth, stem);
            if(!depthPath.empty()) {
                std::cout << "saved " << depthPath << " " << depth->width() << "x" << depth->height()
                          << " depth_scale=" << depth->getValueScale() << " (raw units, multiply by scale for mm)"
                          << std::endl;
            }

            // Depth is safely written by now, so it is fine to spend framesets
            // hunting for colour. Roughly 1 frameset in 5 carries it.
            for(int k = 0; k < 30 && !lastColor; ++k) {
                auto extra = pipeline.waitForFrames(500);
                if(extra != nullptr && extra->colorFrame() != nullptr) {
                    lastColor = extra->colorFrame();
                }
            }
            if(lastColor) {
                const std::string colorPath = save_color_image(lastColor, stem);
                if(!colorPath.empty()) {
                    std::cout << "saved " << colorPath << " " << lastColor->width() << "x"
                              << lastColor->height() << " (not time-synced with depth)" << std::endl;
                }
            }
            else {
                std::cout << "no color frame available; color image not saved" << std::endl;
            }

            if(valid >= min_valid) {
                pipeline.stop();
                return 0;
            }
        }
        catch(std::exception &e) {
            std::cerr << "point cloud failed: " << e.what() << std::endl;
        }
    }
    pipeline.stop();
    std::cerr << "failed to capture dense RGB point cloud, last valid=" << bestValid << std::endl;
    return 4;
}
catch(ob::Error &e) {
    std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs() << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType() << std::endl;
    return 1;
}
catch(std::exception &e) {
    std::cerr << "exception: " << e.what() << std::endl;
    return 1;
}
