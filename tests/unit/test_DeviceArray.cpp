#include <gtest/gtest.h>
#include "core/DeviceArray.hpp"

using namespace lynx;

TEST(DeviceArray, Allocate1D) {
    DeviceArray<double> arr(100);
    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.rows(), 100);
    EXPECT_EQ(arr.cols(), 1);
    EXPECT_NE(arr.data(), nullptr);
}

TEST(DeviceArray, Allocate2D) {
    DeviceArray<double> arr(50, 10);
    EXPECT_EQ(arr.rows(), 50);
    EXPECT_EQ(arr.cols(), 10);
    EXPECT_GE(arr.ld(), 50);  // padded to 8-element boundary
}

TEST(DeviceArray, Alignment) {
    DeviceArray<double> arr(100);
    auto ptr = reinterpret_cast<uintptr_t>(arr.data());
    EXPECT_EQ(ptr % 64, 0u);  // 64-byte aligned

    DeviceArray<double> arr2(50, 10);
    auto ptr2 = reinterpret_cast<uintptr_t>(arr2.data());
    EXPECT_EQ(ptr2 % 64, 0u);
}

TEST(DeviceArray, ZeroInitialized) {
    DeviceArray<double> arr(100);
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(arr(i), 0.0);
}

TEST(DeviceArray, Fill) {
    DeviceArray<double> arr(100);
    arr.fill(3.14);
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(arr(i), 3.14);
}

TEST(DeviceArray, ColumnMajor2D) {
    DeviceArray<double> arr(4, 3);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 4; ++i)
            arr(i, j) = i + j * 10.0;

    EXPECT_DOUBLE_EQ(arr(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(arr(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(arr(0, 1), 10.0);
    EXPECT_DOUBLE_EQ(arr(3, 2), 23.0);

    double* c1 = arr.col(1);
    EXPECT_DOUBLE_EQ(c1[0], 10.0);
    EXPECT_DOUBLE_EQ(c1[1], 11.0);
}

TEST(DeviceArray, MoveSemantics) {
    DeviceArray<double> a(100);
    a.fill(42.0);
    double* ptr = a.data();

    DeviceArray<double> b = std::move(a);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(b.size(), 100);
    EXPECT_DOUBLE_EQ(b(0), 42.0);
}

TEST(DeviceArray, MoveAssignment) {
    DeviceArray<double> a(100);
    a.fill(1.0);

    DeviceArray<double> b(50);
    b = std::move(a);
    EXPECT_EQ(b.size(), 100);
    EXPECT_TRUE(a.empty());
}

TEST(DeviceArray, Clone) {
    DeviceArray<double> a(100);
    for (int i = 0; i < 100; ++i)
        a(i) = static_cast<double>(i);

    DeviceArray<double> b = a.clone();
    EXPECT_EQ(b.size(), 100);
    EXPECT_NE(b.data(), a.data());
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(b(i), static_cast<double>(i));
}

TEST(DeviceArray, Resize) {
    DeviceArray<double> arr(100);
    arr.fill(1.0);
    arr.resize(50);
    EXPECT_EQ(arr.size(), 50);
    EXPECT_DOUBLE_EQ(arr(0), 0.0);  // zeroed after resize
}

TEST(DeviceArray, Resize2D) {
    DeviceArray<double> arr(10, 5);
    arr.resize(20, 3);
    EXPECT_EQ(arr.rows(), 20);
    EXPECT_EQ(arr.cols(), 3);
}

TEST(DeviceArray, LeadingDimensionPadded) {
    DeviceArray<double> arr(10, 5);
    EXPECT_GE(arr.ld(), 10);
    EXPECT_EQ(arr.ld() % 8, 0);  // padded to 8-element boundary
}

TEST(DeviceArray, ComplexType) {
    DeviceArray<std::complex<double>> arr(10);
    arr(0) = {1.0, 2.0};
    EXPECT_DOUBLE_EQ(arr(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(arr(0).imag(), 2.0);
}

TEST(DeviceArray, DeviceTagCPU) {
    DeviceArray<double> arr(100);
    EXPECT_EQ(arr.device(), Device::CPU);
    EXPECT_TRUE(arr.on_cpu());
    EXPECT_FALSE(arr.on_gpu());
}

TEST(DeviceArray, CopyTo) {
    DeviceArray<double> a(100);
    for (int i = 0; i < 100; ++i)
        a(i) = static_cast<double>(i);

    DeviceArray<double> b = a.to(Device::CPU);
    EXPECT_NE(b.data(), a.data());
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(b(i), static_cast<double>(i));
}

TEST(DeviceArray, CopyFrom) {
    DeviceArray<double> a(100);
    DeviceArray<double> b(100);
    for (int i = 0; i < 100; ++i)
        a(i) = static_cast<double>(i * 2);

    b.copy_from(a);
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(b(i), static_cast<double>(i * 2));
}

#ifdef USE_CUDA
TEST(DeviceArray, PinnedMemory) {
    DeviceArray<double> arr(100, Device::CPU_PINNED);
    EXPECT_TRUE(arr.on_cpu());
    EXPECT_TRUE(arr.is_pinned());
    EXPECT_FALSE(arr.on_gpu());

    // Can read/write like normal CPU memory
    arr(0) = 42.0;
    EXPECT_DOUBLE_EQ(arr(0), 42.0);

    // Transfer to GPU
    DeviceArray<double> gpu = arr.to(Device::GPU);
    EXPECT_TRUE(gpu.on_gpu());

    // Transfer back to pinned
    DeviceArray<double> back = gpu.to(Device::CPU_PINNED);
    EXPECT_TRUE(back.is_pinned());
    EXPECT_DOUBLE_EQ(back(0), 42.0);

    // Transfer back to regular CPU
    DeviceArray<double> cpu = gpu.to(Device::CPU);
    EXPECT_TRUE(cpu.on_cpu());
    EXPECT_FALSE(cpu.is_pinned());
    EXPECT_DOUBLE_EQ(cpu(0), 42.0);
}
#endif
