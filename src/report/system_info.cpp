#include "floptic/report.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/utsname.h>

namespace floptic {

static std::string exec_command(const char* cmd) {
    char buffer[256];
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    // Trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

static std::string read_first_line(const char* path) {
    std::ifstream f(path);
    std::string line;
    if (f.is_open()) std::getline(f, line);
    return line;
}

SystemInfo collect_system_info() {
    SystemInfo info;

    // Hostname
    char hbuf[256];
    if (gethostname(hbuf, sizeof(hbuf)) == 0)
        info.hostname = hbuf;
    else
        info.hostname = "unknown";

    // OS info via uname
    struct utsname un;
    if (uname(&un) == 0) {
        info.os_name = un.sysname;
        info.os_release = un.release;
        info.os_arch = un.machine;
    }

    // CPU model from /proc/cpuinfo
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    info.cpu_model = line.substr(pos + 2);
                }
                break;
            }
        }
    }

    // Compiler info (compile-time)
#if defined(__clang__)
    info.compiler_name = "Clang";
    info.compiler_version = __clang_version__;
#elif defined(__GNUC__)
    info.compiler_name = "GCC";
    {
        char vbuf[32];
        snprintf(vbuf, sizeof(vbuf), "%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
        info.compiler_version = vbuf;
    }
#elif defined(_MSC_VER)
    info.compiler_name = "MSVC";
    info.compiler_version = std::to_string(_MSC_VER);
#else
    info.compiler_name = "unknown";
    info.compiler_version = "unknown";
#endif

    // CUDA / GPU info via nvidia-smi (works without linking CUDA)
    {
        std::string smi = exec_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1");
        if (!smi.empty())
            info.nvidia_smi_driver = smi;
    }

    // CUDA toolkit version from nvcc
    {
        std::string nvcc_out = exec_command("nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//'");
        if (!nvcc_out.empty())
            info.cuda_runtime_version = nvcc_out;
    }

    // Loaded modules (HPC environment)
    {
        const char* loadedmodules = std::getenv("LOADEDMODULES");
        if (loadedmodules && strlen(loadedmodules) > 0) {
            std::istringstream ss(loadedmodules);
            std::string mod;
            while (std::getline(ss, mod, ':')) {
                if (!mod.empty())
                    info.loaded_modules.push_back(mod);
            }
        }
    }

    return info;
}

} // namespace floptic
