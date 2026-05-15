#include "cuda_keccak.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace eth_cuda
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;
        constexpr std::size_t kPrivateKeySize = 32;
        constexpr std::size_t kPublicKeySize = 64;
        constexpr std::size_t kAddressSize = 20;
        constexpr int kFieldLimbCount = 8;

        struct ThreadLocalCudaBuffers
        {
            std::uint8_t* devicePrivateKeys = nullptr;
            std::uint8_t* devicePublicKeys = nullptr;
            std::uint8_t* deviceAddresses = nullptr;
            std::size_t privateKeyCapacity = 0;
            std::size_t publicKeyCapacity = 0;
            std::size_t addressCapacity = 0;
            cudaStream_t stream = nullptr;
            bool deviceInitialized = false;

            ~ThreadLocalCudaBuffers()
            {
                if (deviceInitialized) {
                    cudaSetDevice(0);
                }
                if (stream != nullptr) {
                    cudaStreamDestroy(stream);
                }
                if (deviceAddresses != nullptr) {
                    cudaFree(deviceAddresses);
                }
                if (devicePublicKeys != nullptr) {
                    cudaFree(devicePublicKeys);
                }
                if (devicePrivateKeys != nullptr) {
                    cudaFree(devicePrivateKeys);
                }
            }
        };

        thread_local ThreadLocalCudaBuffers gThreadLocalCudaBuffers;

        std::string MakeCudaError(const char* operation, cudaError_t status)
        {
            return std::string(operation) + " failed: " + cudaGetErrorString(status);
        }

        bool EnsureCudaDevice(ThreadLocalCudaBuffers& buffers, std::string& error)
        {
            if (buffers.deviceInitialized) {
                return true;
            }

            const cudaError_t status = cudaSetDevice(0);
            if (status != cudaSuccess) {
                error = MakeCudaError("cudaSetDevice", status);
                return false;
            }

            buffers.deviceInitialized = true;
            return true;
        }

        bool EnsureCudaStream(ThreadLocalCudaBuffers& buffers, std::string& error)
        {
            if (buffers.stream != nullptr) {
                return true;
            }

            const cudaError_t status = cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking);
            if (status != cudaSuccess) {
                error = MakeCudaError("cudaStreamCreateWithFlags", status);
                return false;
            }

            return true;
        }

        bool EnsureCapacity(
            std::uint8_t*& deviceBuffer,
            std::size_t& currentCapacity,
            std::size_t requiredCapacity,
            std::string& error)
        {
            if (currentCapacity >= requiredCapacity) {
                return true;
            }

            if (deviceBuffer != nullptr) {
                const cudaError_t freeStatus = cudaFree(deviceBuffer);
                if (freeStatus != cudaSuccess) {
                    error = MakeCudaError("cudaFree", freeStatus);
                    deviceBuffer = nullptr;
                    currentCapacity = 0;
                    return false;
                }

                deviceBuffer = nullptr;
                currentCapacity = 0;
            }

            const cudaError_t allocStatus = cudaMalloc(reinterpret_cast<void**>(&deviceBuffer), requiredCapacity);
            if (allocStatus != cudaSuccess) {
                error = MakeCudaError("cudaMalloc", allocStatus);
                deviceBuffer = nullptr;
                currentCapacity = 0;
                return false;
            }

            currentCapacity = requiredCapacity;
            return true;
        }

        __device__ __constant__ std::uint32_t kFieldPrime[kFieldLimbCount] = {
            0xFFFFFC2FU, 0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU,
            0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
        };

        __device__ __constant__ std::uint32_t kFieldPrimeMinusTwo[kFieldLimbCount] = {
            0xFFFFFC2DU, 0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU,
            0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
        };

        __device__ __constant__ std::uint32_t kGeneratorX[kFieldLimbCount] = {
            0x16F81798U, 0x59F2815BU, 0x2DCE28D9U, 0x029BFCDBU,
            0xCE870B07U, 0x55A06295U, 0xF9DCBBACU, 0x79BE667EU
        };

        __device__ __constant__ std::uint32_t kGeneratorY[kFieldLimbCount] = {
            0xFB10D4B8U, 0x9C47D08FU, 0xA6855419U, 0xFD17B448U,
            0x0E1108A8U, 0x5DA4FBFCU, 0x26A3C465U, 0x483ADA77U
        };

        __device__ __constant__ std::uint64_t kRoundConstants[24] = {
            0x0000000000000001ULL, 0x0000000000008082ULL,
            0x800000000000808aULL, 0x8000000080008000ULL,
            0x000000000000808bULL, 0x0000000080000001ULL,
            0x8000000080008081ULL, 0x8000000000008009ULL,
            0x000000000000008aULL, 0x0000000000000088ULL,
            0x0000000080008009ULL, 0x000000008000000aULL,
            0x000000008000808bULL, 0x800000000000008bULL,
            0x8000000000008089ULL, 0x8000000000008003ULL,
            0x8000000000008002ULL, 0x8000000000000080ULL,
            0x000000000000800aULL, 0x800000008000000aULL,
            0x8000000080008081ULL, 0x8000000000008080ULL,
            0x0000000080000001ULL, 0x8000000080008008ULL
        };

        __device__ __constant__ int kRotationOffsets[24] = {
            1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
            27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
        };

        __device__ __constant__ int kPiLane[24] = {
            10,  7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
            15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
        };

        struct JacobianPoint
        {
            std::uint32_t x[kFieldLimbCount];
            std::uint32_t y[kFieldLimbCount];
            std::uint32_t z[kFieldLimbCount];
            bool infinity;
        };

        __device__ __forceinline__ void SetZero(std::uint32_t value[kFieldLimbCount])
        {
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                value[i] = 0;
            }
        }

        __device__ __forceinline__ void SetOne(std::uint32_t value[kFieldLimbCount])
        {
            value[0] = 1;
            #pragma unroll
            for (int i = 1; i < kFieldLimbCount; ++i) {
                value[i] = 0;
            }
        }

        __device__ __forceinline__ void CopyField(
            std::uint32_t dst[kFieldLimbCount],
            const std::uint32_t src[kFieldLimbCount])
        {
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                dst[i] = src[i];
            }
        }

        __device__ __forceinline__ bool IsZeroField(const std::uint32_t value[kFieldLimbCount])
        {
            std::uint32_t accum = 0;
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                accum |= value[i];
            }
            return accum == 0;
        }

        __device__ __forceinline__ int CompareField(
            const std::uint32_t lhs[kFieldLimbCount],
            const std::uint32_t rhs[kFieldLimbCount])
        {
            for (int i = kFieldLimbCount - 1; i >= 0; --i) {
                if (lhs[i] < rhs[i]) {
                    return -1;
                }
                if (lhs[i] > rhs[i]) {
                    return 1;
                }
            }
            return 0;
        }

        __device__ __forceinline__ std::uint32_t AddRaw(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t lhs[kFieldLimbCount],
            const std::uint32_t rhs[kFieldLimbCount])
        {
            std::uint64_t carry = 0;
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                const std::uint64_t sum =
                    static_cast<std::uint64_t>(lhs[i]) +
                    static_cast<std::uint64_t>(rhs[i]) +
                    carry;
                out[i] = static_cast<std::uint32_t>(sum);
                carry = sum >> 32;
            }
            return static_cast<std::uint32_t>(carry);
        }

        __device__ __forceinline__ std::uint32_t SubRaw(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t lhs[kFieldLimbCount],
            const std::uint32_t rhs[kFieldLimbCount])
        {
            std::uint64_t borrow = 0;
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                const std::uint64_t minuend = static_cast<std::uint64_t>(lhs[i]);
                const std::uint64_t subtrahend = static_cast<std::uint64_t>(rhs[i]) + borrow;
                out[i] = static_cast<std::uint32_t>(minuend - subtrahend);
                borrow = (minuend < subtrahend) ? 1ULL : 0ULL;
            }
            return static_cast<std::uint32_t>(borrow);
        }

        __device__ __forceinline__ void FoldCarryConstant(std::uint32_t out[kFieldLimbCount])
        {
            std::uint64_t sum = static_cast<std::uint64_t>(out[0]) + 977ULL;
            out[0] = static_cast<std::uint32_t>(sum);
            std::uint64_t carry = sum >> 32;

            sum = static_cast<std::uint64_t>(out[1]) + 1ULL + carry;
            out[1] = static_cast<std::uint32_t>(sum);
            carry = sum >> 32;

            for (int i = 2; i < kFieldLimbCount && carry != 0; ++i) {
                sum = static_cast<std::uint64_t>(out[i]) + carry;
                out[i] = static_cast<std::uint32_t>(sum);
                carry = sum >> 32;
            }
        }

        __device__ __forceinline__ void FoldBorrowConstant(std::uint32_t out[kFieldLimbCount])
        {
            std::uint64_t borrow = 977ULL;
            for (int i = 0; i < kFieldLimbCount && borrow != 0; ++i) {
                const std::uint64_t minuend = static_cast<std::uint64_t>(out[i]);
                const std::uint64_t subtrahend = borrow & 0xFFFFFFFFULL;
                out[i] = static_cast<std::uint32_t>(minuend - subtrahend);
                borrow = (minuend < subtrahend) ? 1ULL : 0ULL;
            }

            borrow = 1ULL;
            for (int i = 1; i < kFieldLimbCount && borrow != 0; ++i) {
                const std::uint64_t minuend = static_cast<std::uint64_t>(out[i]);
                out[i] = static_cast<std::uint32_t>(minuend - borrow);
                borrow = (minuend == 0) ? 1ULL : 0ULL;
            }
        }

        __device__ __forceinline__ void NormalizeField(std::uint32_t out[kFieldLimbCount])
        {
            while (CompareField(out, kFieldPrime) >= 0) {
                SubRaw(out, out, kFieldPrime);
            }
        }

        __device__ __forceinline__ void FieldAdd(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t lhs[kFieldLimbCount],
            const std::uint32_t rhs[kFieldLimbCount])
        {
            const std::uint32_t carry = AddRaw(out, lhs, rhs);
            if (carry != 0) {
                FoldCarryConstant(out);
            }
            NormalizeField(out);
        }

        __device__ __forceinline__ void FieldSub(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t lhs[kFieldLimbCount],
            const std::uint32_t rhs[kFieldLimbCount])
        {
            const std::uint32_t borrow = SubRaw(out, lhs, rhs);
            if (borrow != 0) {
                FoldBorrowConstant(out);
            }
        }

        __device__ __forceinline__ void FieldDouble(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t value[kFieldLimbCount])
        {
            FieldAdd(out, value, value);
        }

        __device__ __forceinline__ void FieldTriple(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t value[kFieldLimbCount])
        {
            std::uint32_t doubled[kFieldLimbCount];
            FieldDouble(doubled, value);
            FieldAdd(out, doubled, value);
        }

        __device__ __forceinline__ void FieldMulBy4(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t value[kFieldLimbCount])
        {
            FieldDouble(out, value);
            FieldDouble(out, out);
        }

        __device__ __forceinline__ void FieldMulBy8(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t value[kFieldLimbCount])
        {
            FieldMulBy4(out, value);
            FieldDouble(out, out);
        }

        __device__ __forceinline__ void AddWord(
            std::uint32_t* words,
            int wordCount,
            int position,
            std::uint32_t value)
        {
            std::uint64_t carry = value;
            int index = position;
            while (carry != 0 && index < wordCount) {
                const std::uint64_t sum = static_cast<std::uint64_t>(words[index]) + (carry & 0xFFFFFFFFULL);
                words[index] = static_cast<std::uint32_t>(sum);
                carry = (carry >> 32) + (sum >> 32);
                ++index;
            }
        }

        __device__ __forceinline__ void AddMulWord(
            std::uint32_t* words,
            int wordCount,
            int position,
            std::uint32_t value,
            std::uint32_t multiplier)
        {
            std::uint64_t carry = static_cast<std::uint64_t>(value) * static_cast<std::uint64_t>(multiplier);
            int index = position;
            while (carry != 0 && index < wordCount) {
                const std::uint64_t sum = static_cast<std::uint64_t>(words[index]) + (carry & 0xFFFFFFFFULL);
                words[index] = static_cast<std::uint32_t>(sum);
                carry = (carry >> 32) + (sum >> 32);
                ++index;
            }
        }

        __device__ __forceinline__ void FieldReduceProduct(
            const std::uint32_t product[16],
            std::uint32_t out[kFieldLimbCount])
        {
            std::uint32_t reduced[18] = {};
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                reduced[i] = product[i];
            }

            for (;;) {
                int top = 17;
                while (top >= kFieldLimbCount && reduced[top] == 0) {
                    --top;
                }
                if (top < kFieldLimbCount) {
                    break;
                }

                const std::uint32_t high = reduced[top];
                reduced[top] = 0;
                if (high == 0) {
                    continue;
                }

                AddMulWord(reduced, 18, top - kFieldLimbCount, high, 977U);
                AddWord(reduced, 18, top - (kFieldLimbCount - 1), high);
            }

            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                out[i] = reduced[i];
            }

            NormalizeField(out);
        }

        __device__ __forceinline__ void FieldMul(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t lhs[kFieldLimbCount],
            const std::uint32_t rhs[kFieldLimbCount])
        {
            std::uint32_t product[16] = {};
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                std::uint64_t carry = 0;
                #pragma unroll
                for (int j = 0; j < kFieldLimbCount; ++j) {
                    const int k = i + j;
                    const std::uint64_t sum =
                        static_cast<std::uint64_t>(product[k]) +
                        (static_cast<std::uint64_t>(lhs[i]) * static_cast<std::uint64_t>(rhs[j])) +
                        carry;
                    product[k] = static_cast<std::uint32_t>(sum);
                    carry = sum >> 32;
                }

                int index = i + kFieldLimbCount;
                while (carry != 0) {
                    const std::uint64_t sum = static_cast<std::uint64_t>(product[index]) + carry;
                    product[index] = static_cast<std::uint32_t>(sum);
                    carry = sum >> 32;
                    ++index;
                }
            }

            FieldReduceProduct(product, out);
        }

        __device__ __forceinline__ void FieldSquare(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t value[kFieldLimbCount])
        {
            FieldMul(out, value, value);
        }

        __device__ void FieldInverse(
            std::uint32_t out[kFieldLimbCount],
            const std::uint32_t value[kFieldLimbCount])
        {
            if (IsZeroField(value)) {
                SetZero(out);
                return;
            }

            std::uint32_t result[kFieldLimbCount];
            SetOne(result);

            for (int bit = 255; bit >= 0; --bit) {
                FieldSquare(result, result);
                const std::uint32_t limb = kFieldPrimeMinusTwo[bit >> 5];
                const std::uint32_t mask = 1U << (bit & 31);
                if ((limb & mask) != 0) {
                    FieldMul(result, result, value);
                }
            }

            CopyField(out, result);
        }

        __device__ __forceinline__ void SetInfinity(JacobianPoint& point)
        {
            SetZero(point.x);
            SetZero(point.y);
            SetZero(point.z);
            point.infinity = true;
        }

        __device__ __forceinline__ void SetGenerator(JacobianPoint& point)
        {
            CopyField(point.x, kGeneratorX);
            CopyField(point.y, kGeneratorY);
            SetOne(point.z);
            point.infinity = false;
        }

        __device__ void PointDouble(JacobianPoint& point)
        {
            if (point.infinity || IsZeroField(point.y)) {
                SetInfinity(point);
                return;
            }

            std::uint32_t xx[kFieldLimbCount];
            std::uint32_t yy[kFieldLimbCount];
            std::uint32_t yyyy[kFieldLimbCount];
            std::uint32_t s[kFieldLimbCount];
            std::uint32_t m[kFieldLimbCount];
            std::uint32_t x3[kFieldLimbCount];
            std::uint32_t y3[kFieldLimbCount];
            std::uint32_t z3[kFieldLimbCount];
            std::uint32_t temp[kFieldLimbCount];

            FieldSquare(xx, point.x);
            FieldSquare(yy, point.y);
            FieldSquare(yyyy, yy);

            FieldMul(s, point.x, yy);
            FieldMulBy4(s, s);

            FieldTriple(m, xx);

            FieldSquare(x3, m);
            FieldDouble(temp, s);
            FieldSub(x3, x3, temp);

            FieldSub(temp, s, x3);
            FieldMul(y3, m, temp);
            FieldMulBy8(temp, yyyy);
            FieldSub(y3, y3, temp);

            FieldMul(z3, point.y, point.z);
            FieldDouble(z3, z3);

            CopyField(point.x, x3);
            CopyField(point.y, y3);
            CopyField(point.z, z3);
            point.infinity = false;
        }

        __device__ void PointAddGenerator(JacobianPoint& point)
        {
            if (point.infinity) {
                SetGenerator(point);
                return;
            }

            std::uint32_t z1z1[kFieldLimbCount];
            std::uint32_t u2[kFieldLimbCount];
            std::uint32_t s2[kFieldLimbCount];
            std::uint32_t h[kFieldLimbCount];
            std::uint32_t hh[kFieldLimbCount];
            std::uint32_t i[kFieldLimbCount];
            std::uint32_t j[kFieldLimbCount];
            std::uint32_t r[kFieldLimbCount];
            std::uint32_t v[kFieldLimbCount];
            std::uint32_t x3[kFieldLimbCount];
            std::uint32_t y3[kFieldLimbCount];
            std::uint32_t z3[kFieldLimbCount];
            std::uint32_t temp[kFieldLimbCount];
            std::uint32_t deltaY[kFieldLimbCount];

            FieldSquare(z1z1, point.z);
            FieldMul(u2, kGeneratorX, z1z1);
            FieldMul(temp, z1z1, point.z);
            FieldMul(s2, kGeneratorY, temp);

            FieldSub(h, u2, point.x);
            FieldSub(deltaY, s2, point.y);

            if (IsZeroField(h)) {
                if (IsZeroField(deltaY)) {
                    PointDouble(point);
                } else {
                    SetInfinity(point);
                }
                return;
            }

            FieldSquare(hh, h);
            FieldMulBy4(i, hh);
            FieldMul(j, h, i);
            FieldDouble(r, deltaY);
            FieldMul(v, point.x, i);

            FieldSquare(x3, r);
            FieldDouble(temp, v);
            FieldSub(x3, x3, j);
            FieldSub(x3, x3, temp);

            FieldSub(temp, v, x3);
            FieldMul(y3, r, temp);
            FieldMul(temp, point.y, j);
            FieldDouble(temp, temp);
            FieldSub(y3, y3, temp);

            FieldAdd(z3, point.z, h);
            FieldSquare(z3, z3);
            FieldSub(z3, z3, z1z1);
            FieldSub(z3, z3, hh);

            CopyField(point.x, x3);
            CopyField(point.y, y3);
            CopyField(point.z, z3);
            point.infinity = false;
        }

        __device__ __forceinline__ int GetScalarBit(const std::uint8_t* scalar, int bitIndex)
        {
            return (scalar[31 - (bitIndex >> 3)] >> (bitIndex & 7)) & 0x01;
        }

        __device__ void ScalarMultiplyGenerator(
            const std::uint8_t* scalar,
            std::uint32_t outX[kFieldLimbCount],
            std::uint32_t outY[kFieldLimbCount],
            bool& isInfinity)
        {
            JacobianPoint point;
            SetInfinity(point);

            for (int bit = 255; bit >= 0; --bit) {
                if (!point.infinity) {
                    PointDouble(point);
                }
                if (GetScalarBit(scalar, bit) != 0) {
                    PointAddGenerator(point);
                }
            }

            if (point.infinity || IsZeroField(point.z)) {
                isInfinity = true;
                SetZero(outX);
                SetZero(outY);
                return;
            }

            std::uint32_t zInv[kFieldLimbCount];
            std::uint32_t zInv2[kFieldLimbCount];
            std::uint32_t zInv3[kFieldLimbCount];

            FieldInverse(zInv, point.z);
            FieldSquare(zInv2, zInv);
            FieldMul(zInv3, zInv2, zInv);
            FieldMul(outX, point.x, zInv2);
            FieldMul(outY, point.y, zInv3);
            isInfinity = false;
        }

        __device__ __forceinline__ void FieldToBytesBigEndian(
            const std::uint32_t value[kFieldLimbCount],
            std::uint8_t* output)
        {
            #pragma unroll
            for (int i = 0; i < kFieldLimbCount; ++i) {
                const std::uint32_t limb = value[kFieldLimbCount - 1 - i];
                output[(i * 4) + 0] = static_cast<std::uint8_t>((limb >> 24) & 0xFF);
                output[(i * 4) + 1] = static_cast<std::uint8_t>((limb >> 16) & 0xFF);
                output[(i * 4) + 2] = static_cast<std::uint8_t>((limb >> 8) & 0xFF);
                output[(i * 4) + 3] = static_cast<std::uint8_t>(limb & 0xFF);
            }
        }

        __device__ __forceinline__ std::uint64_t LoadLE64(const std::uint8_t* src)
        {
            std::uint64_t value = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                value |= static_cast<std::uint64_t>(src[i]) << (8 * i);
            }
            return value;
        }

        __device__ __forceinline__ void StoreLE64(std::uint8_t* dst, std::uint64_t value)
        {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dst[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
            }
        }

        __device__ __forceinline__ std::uint64_t RotL64(std::uint64_t value, int shift)
        {
            return (value << shift) | (value >> (64 - shift));
        }

        __device__ void KeccakF1600(std::uint64_t state[25])
        {
            for (int round = 0; round < 24; ++round) {
                std::uint64_t c[5] = {};
                #pragma unroll
                for (int x = 0; x < 5; ++x) {
                    c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
                }

                std::uint64_t d[5] = {};
                #pragma unroll
                for (int x = 0; x < 5; ++x) {
                    d[x] = c[(x + 4) % 5] ^ RotL64(c[(x + 1) % 5], 1);
                }

                #pragma unroll
                for (int x = 0; x < 5; ++x) {
                    #pragma unroll
                    for (int y = 0; y < 25; y += 5) {
                        state[x + y] ^= d[x];
                    }
                }

                std::uint64_t current = state[1];
                #pragma unroll
                for (int i = 0; i < 24; ++i) {
                    const int lane = kPiLane[i];
                    const std::uint64_t tmp = state[lane];
                    state[lane] = RotL64(current, kRotationOffsets[i]);
                    current = tmp;
                }

                #pragma unroll
                for (int y = 0; y < 25; y += 5) {
                    std::uint64_t row[5] = {};
                    #pragma unroll
                    for (int x = 0; x < 5; ++x) {
                        row[x] = state[y + x];
                    }
                    #pragma unroll
                    for (int x = 0; x < 5; ++x) {
                        state[y + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
                    }
                }

                state[0] ^= kRoundConstants[round];
            }
        }

        __device__ __forceinline__ void StoreAddressBytes(std::uint8_t* output, const std::uint64_t state[25])
        {
            const std::uint64_t lane1 = state[1];
            output[0] = static_cast<std::uint8_t>((lane1 >> 32) & 0xFF);
            output[1] = static_cast<std::uint8_t>((lane1 >> 40) & 0xFF);
            output[2] = static_cast<std::uint8_t>((lane1 >> 48) & 0xFF);
            output[3] = static_cast<std::uint8_t>((lane1 >> 56) & 0xFF);
            StoreLE64(output + 4, state[2]);
            StoreLE64(output + 12, state[3]);
        }

        __device__ __forceinline__ void KeccakPublicKeyToAddress(
            const std::uint8_t* publicKeyBytes,
            std::uint8_t* output)
        {
            std::uint64_t state[25] = {};
            #pragma unroll
            for (int lane = 0; lane < 8; ++lane) {
                state[lane] = LoadLE64(publicKeyBytes + (lane * 8));
            }

            state[8] ^= 0x01ULL;
            state[16] ^= 0x8000000000000000ULL;
            KeccakF1600(state);
            StoreAddressBytes(output, state);
        }

        __global__ __launch_bounds__(kThreadsPerBlock) void KeccakAddressKernel(
            const std::uint8_t* publicKeysNoPrefix,
            std::uint8_t* outAddresses,
            std::size_t keyCount)
        {
            const std::size_t idx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
            if (idx >= keyCount) {
                return;
            }

            const std::uint8_t* input = publicKeysNoPrefix + (idx * kPublicKeySize);
            std::uint8_t* output = outAddresses + (idx * kAddressSize);
            KeccakPublicKeyToAddress(input, output);
        }

        __global__ __launch_bounds__(kThreadsPerBlock) void PrivateKeyToAddressKernel(
            const std::uint8_t* privateKeys,
            std::uint8_t* outAddresses,
            std::size_t keyCount)
        {
            const std::size_t idx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
            if (idx >= keyCount) {
                return;
            }

            const std::uint8_t* scalar = privateKeys + (idx * kPrivateKeySize);
            std::uint8_t* output = outAddresses + (idx * kAddressSize);

            std::uint32_t x[kFieldLimbCount];
            std::uint32_t y[kFieldLimbCount];
            bool isInfinity = false;
            ScalarMultiplyGenerator(scalar, x, y, isInfinity);

            if (isInfinity) {
                #pragma unroll
                for (int i = 0; i < static_cast<int>(kAddressSize); ++i) {
                    output[i] = 0;
                }
                return;
            }

            std::uint8_t publicKey[kPublicKeySize];
            FieldToBytesBigEndian(x, publicKey);
            FieldToBytesBigEndian(y, publicKey + 32);
            KeccakPublicKeyToAddress(publicKey, output);
        }
    } // namespace

    CudaDeviceInfo QueryCudaDeviceInfo()
    {
        CudaDeviceInfo info;

        int deviceCount = 0;
        const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
        if (countStatus != cudaSuccess) {
            info.error = MakeCudaError("cudaGetDeviceCount", countStatus);
            return info;
        }

        info.deviceCount = deviceCount;
        if (deviceCount < 1) {
            info.error = "No CUDA GPU detected.";
            return info;
        }

        cudaDeviceProp deviceProperties = {};
        const cudaError_t propsStatus = cudaGetDeviceProperties(&deviceProperties, 0);
        if (propsStatus != cudaSuccess) {
            info.error = MakeCudaError("cudaGetDeviceProperties", propsStatus);
            return info;
        }

        info.available = true;
        info.selectedDevice = 0;
        info.computeMajor = deviceProperties.major;
        info.computeMinor = deviceProperties.minor;
        info.name = deviceProperties.name;
        return info;
    }

    bool ComputeEthereumAddressesFromPrivateKeysCuda(
        const std::uint8_t* privateKeys,
        std::size_t keyCount,
        std::uint8_t* outAddressBytes,
        std::string& error)
    {
        error.clear();

        if (privateKeys == nullptr) {
            error = "Private key input buffer is null.";
            return false;
        }
        if (outAddressBytes == nullptr) {
            error = "Address output buffer is null.";
            return false;
        }
        if (keyCount == 0) {
            return true;
        }

        ThreadLocalCudaBuffers& buffers = gThreadLocalCudaBuffers;
        if (!EnsureCudaDevice(buffers, error)) {
            return false;
        }
        if (!EnsureCudaStream(buffers, error)) {
            return false;
        }

        const std::size_t inputBytes = keyCount * kPrivateKeySize;
        const std::size_t outputBytes = keyCount * kAddressSize;

        if (!EnsureCapacity(buffers.devicePrivateKeys, buffers.privateKeyCapacity, inputBytes, error)) {
            return false;
        }
        if (!EnsureCapacity(buffers.deviceAddresses, buffers.addressCapacity, outputBytes, error)) {
            return false;
        }

        cudaError_t status = cudaMemcpyAsync(
            buffers.devicePrivateKeys,
            privateKeys,
            inputBytes,
            cudaMemcpyHostToDevice,
            buffers.stream);
        if (status != cudaSuccess) {
            error = MakeCudaError("cudaMemcpyAsync(H2D privateKeys)", status);
            return false;
        }

        const int blockCount = static_cast<int>((keyCount + static_cast<std::size_t>(kThreadsPerBlock) - 1) / static_cast<std::size_t>(kThreadsPerBlock));
        PrivateKeyToAddressKernel<<<blockCount, kThreadsPerBlock, 0, buffers.stream>>>(
            buffers.devicePrivateKeys,
            buffers.deviceAddresses,
            keyCount);

        status = cudaPeekAtLastError();
        if (status != cudaSuccess) {
            error = MakeCudaError("launch PrivateKeyToAddressKernel", status);
            return false;
        }

        status = cudaMemcpyAsync(
            outAddressBytes,
            buffers.deviceAddresses,
            outputBytes,
            cudaMemcpyDeviceToHost,
            buffers.stream);
        if (status != cudaSuccess) {
            error = MakeCudaError("cudaMemcpyAsync(D2H addresses)", status);
            return false;
        }

        status = cudaStreamSynchronize(buffers.stream);
        if (status != cudaSuccess) {
            error = MakeCudaError("cudaStreamSynchronize", status);
            return false;
        }

        return true;
    }

    bool ComputeEthereumAddressHashesCuda(
        const std::uint8_t* publicKeysNoPrefix,
        std::size_t keyCount,
        std::uint8_t* outAddressBytes,
        std::string& error)
    {
        error.clear();

        if (publicKeysNoPrefix == nullptr) {
            error = "Public key input buffer is null.";
            return false;
        }
        if (outAddressBytes == nullptr) {
            error = "Address output buffer is null.";
            return false;
        }
        if (keyCount == 0) {
            return true;
        }

        ThreadLocalCudaBuffers& buffers = gThreadLocalCudaBuffers;
        if (!EnsureCudaDevice(buffers, error)) {
            return false;
        }
        if (!EnsureCudaStream(buffers, error)) {
            return false;
        }

        const std::size_t inputBytes = keyCount * kPublicKeySize;
        const std::size_t outputBytes = keyCount * kAddressSize;

        if (!EnsureCapacity(buffers.devicePublicKeys, buffers.publicKeyCapacity, inputBytes, error)) {
            return false;
        }
        if (!EnsureCapacity(buffers.deviceAddresses, buffers.addressCapacity, outputBytes, error)) {
            return false;
        }

        cudaError_t status = cudaMemcpyAsync(
            buffers.devicePublicKeys,
            publicKeysNoPrefix,
            inputBytes,
            cudaMemcpyHostToDevice,
            buffers.stream);
        if (status != cudaSuccess) {
            error = MakeCudaError("cudaMemcpyAsync(H2D publicKeys)", status);
            return false;
        }

        const int blockCount = static_cast<int>((keyCount + static_cast<std::size_t>(kThreadsPerBlock) - 1) / static_cast<std::size_t>(kThreadsPerBlock));
        KeccakAddressKernel<<<blockCount, kThreadsPerBlock, 0, buffers.stream>>>(
            buffers.devicePublicKeys,
            buffers.deviceAddresses,
            keyCount);

        status = cudaPeekAtLastError();
        if (status != cudaSuccess) {
            error = MakeCudaError("launch KeccakAddressKernel", status);
            return false;
        }

        status = cudaMemcpyAsync(
            outAddressBytes,
            buffers.deviceAddresses,
            outputBytes,
            cudaMemcpyDeviceToHost,
            buffers.stream);
        if (status != cudaSuccess) {
            error = MakeCudaError("cudaMemcpyAsync(D2H addresses)", status);
            return false;
        }

        status = cudaStreamSynchronize(buffers.stream);
        if (status != cudaSuccess) {
            error = MakeCudaError("cudaStreamSynchronize", status);
            return false;
        }

        return true;
    }
}
