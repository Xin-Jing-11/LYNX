#include "io/ResultWriter.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace lynx {

namespace {
std::string iso8601_utc_now() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &time);
#else
    gmtime_r(&time, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}
} // namespace

void ResultWriter::write(const std::string& path) const {
    nlohmann::json j;
    j["lynx_version"]  = lynx_version_;
    j["test_name"]     = test_name_;
    j["input_file"]    = input_file_;
    j["timestamp_utc"] = iso8601_utc_now();
    j["device"]        = device_;
    j["mpi_ranks"]     = mpi_ranks_;

    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("ResultWriter: cannot open " + path);
    out << j.dump(2);
}

} // namespace lynx
