#pragma once

#include <string>

namespace lynx {

class ResultWriter {
public:
    ResultWriter() = default;

    void set_test_name(const std::string& name)    { test_name_ = name; }
    void set_input_file(const std::string& path)   { input_file_ = path; }
    void set_device(const std::string& dev)        { device_ = dev; }
    void set_mpi_ranks(int n)                      { mpi_ranks_ = n; }
    void set_lynx_version(const std::string& v)    { lynx_version_ = v; }

    /// Serialise the accumulated state to `path` as JSON.
    /// Creates parent directories if missing.
    void write(const std::string& path) const;

private:
    std::string test_name_;
    std::string input_file_;
    std::string device_ = "CPU";
    int         mpi_ranks_ = 1;
    std::string lynx_version_;
};

} // namespace lynx
