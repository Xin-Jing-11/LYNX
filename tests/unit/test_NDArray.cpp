#include <gtest/gtest.h>
#include "core/NDArray.hpp"

using namespace lynx;

TEST(NDArray, Allocate1D) {
    NDArray<double> arr(100);
    EXPECT_EQ(arr.size(), 100);
    EXPECT_EQ(arr.ndim(), 1);
    EXPECT_EQ(arr.rows(), 100);
    EXPECT_NE(arr.data(), nullptr);
}

TEST(NDArray, Allocate2D) {
    NDArray<double> arr(50, 10);
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.rows(), 50);
    EXPECT_EQ(arr.cols(), 10);
    EXPECT_EQ(arr.size(), 500);
    EXPECT_GE(arr.ld(), 50);  // may be padded
}

TEST(NDArray, Allocate3D) {
    NDArray<double> arr(10, 20, 30);
    EXPECT_EQ(arr.ndim(), 3);
    EXPECT_EQ(arr.rows(), 10);
    EXPECT_EQ(arr.cols(), 20);
    EXPECT_EQ(arr.depth(), 30);
    EXPECT_EQ(arr.size(), 6000);
}

TEST(NDArray, Alignment) {
    NDArray<double> arr(100);
    EXPECT_TRUE(arr.is_aligned());

    NDArray<double> arr2(50, 10);
    EXPECT_TRUE(arr2.is_aligned());
}

TEST(NDArray, ZeroInitialized) {
    NDArray<double> arr(100);
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(arr(i), 0.0);
}

TEST(NDArray, Fill) {
    NDArray<double> arr(100);
    arr.fill(3.14);
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(arr(i), 3.14);
}

TEST(NDArray, ColumnMajor2D) {
    NDArray<double> arr(4, 3);
    // Set values
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 4; ++i)
            arr(i, j) = i + j * 10.0;

    // Verify column-major layout
    EXPECT_DOUBLE_EQ(arr(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(arr(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(arr(0, 1), 10.0);
    EXPECT_DOUBLE_EQ(arr(3, 2), 23.0);

    // Column pointer
    double* c1 = arr.col(1);
    EXPECT_DOUBLE_EQ(c1[0], 10.0);
    EXPECT_DOUBLE_EQ(c1[1], 11.0);
}

TEST(NDArray, MoveSemantics) {
    NDArray<double> a(100);
    a.fill(42.0);
    double* ptr = a.data();

    NDArray<double> b = std::move(a);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(b.size(), 100);
    EXPECT_DOUBLE_EQ(b(0), 42.0);
}

TEST(NDArray, MoveAssignment) {
    NDArray<double> a(100);
    a.fill(1.0);

    NDArray<double> b(50);
    b = std::move(a);
    EXPECT_EQ(b.size(), 100);
    EXPECT_TRUE(a.empty());
}

TEST(NDArray, Clone) {
    NDArray<double> a(100);
    for (int i = 0; i < 100; ++i)
        a(i) = static_cast<double>(i);

    NDArray<double> b = a.clone();
    EXPECT_EQ(b.size(), 100);
    EXPECT_NE(b.data(), a.data());
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(b(i), static_cast<double>(i));
}

TEST(NDArray, Resize) {
    NDArray<double> arr(100);
    arr.fill(1.0);
    arr.resize(50);
    EXPECT_EQ(arr.size(), 50);
    EXPECT_DOUBLE_EQ(arr(0), 0.0);  // zeroed after resize
}

TEST(NDArray, Resize2D) {
    NDArray<double> arr(10, 5);
    arr.resize(20, 3);
    EXPECT_EQ(arr.rows(), 20);
    EXPECT_EQ(arr.cols(), 3);
}

TEST(NDArray, ThrowsOnInvalidDimension) {
    EXPECT_THROW(NDArray<double>(-1), std::invalid_argument);
    EXPECT_THROW(NDArray<double>(0), std::invalid_argument);
}

TEST(NDArray, Index3D) {
    NDArray<double> arr(3, 4, 5);
    arr(1, 2, 3) = 99.0;
    EXPECT_DOUBLE_EQ(arr(1, 2, 3), 99.0);
}

TEST(NDArray, LeadingDimensionPadded) {
    // For 2D arrays, ld should be >= rows
    // Note: padding is currently disabled (ld == rows) to avoid stride bugs
    // with EigenSolver. When re-enabled, ld should be multiple of 8.
    NDArray<double> arr(10, 5);
    EXPECT_GE(arr.ld(), 10);
}

TEST(NDArray, ComplexType) {
    NDArray<std::complex<double>> arr(10);
    arr(0) = {1.0, 2.0};
    EXPECT_DOUBLE_EQ(arr(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(arr(0).imag(), 2.0);
}
