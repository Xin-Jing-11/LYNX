#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>

#include "io/ResultWriter.hpp"

using namespace lynx;
using nlohmann::json;

TEST(ResultWriter, EmitsMinimalValidJson) {
    ResultWriter rw;
    rw.set_test_name("Unit.Minimal");
    rw.set_input_file("dummy.json");
    rw.set_device("CPU");
    rw.set_mpi_ranks(1);
    rw.set_lynx_version("test");

    const std::string path = "/tmp/lynx_rw_test_minimal.json";
    std::remove(path.c_str());
    rw.write(path);

    std::ifstream f(path);
    ASSERT_TRUE(f.good()) << "File not created: " << path;
    json j;
    f >> j;
    EXPECT_EQ(j.at("test_name").get<std::string>(), "Unit.Minimal");
    EXPECT_EQ(j.at("input_file").get<std::string>(), "dummy.json");
    EXPECT_EQ(j.at("device").get<std::string>(), "CPU");
    EXPECT_EQ(j.at("mpi_ranks").get<int>(), 1);
    EXPECT_EQ(j.at("lynx_version").get<std::string>(), "test");
    EXPECT_TRUE(j.contains("timestamp_utc"));
}
