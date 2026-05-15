#define WIN32_LEAN_AND_MEAN
#define OPENSSL_SUPPRESS_DEPRECATED
#define NOMINMAX

// =========================
// Runtime configuration
// =========================
// Set to empty string ("") to disable forced address mode.
#define CONFIG_FORCE_ADDRESS ""
// Example: "0x3bf04228de5e6c54ba021e1fe843fc2e34a2285c" <- Bitmine coinbase address

#define CONFIG_RPC_URL L"http://127.0.0.1:8545"
#define CONFIG_WORKER_THREADS 80
#define CONFIG_WALLETS_PER_BATCH 200
#define CONFIG_INFLIGHT_REQUESTS 3
#define CONFIG_STATS_INTERVAL_SECONDS 10
#define CONFIG_REQUEST_TIMEOUT_MS 15000
#define CONFIG_FOUND_WALLETS_FILE "wallets.txt"
#define CONFIG_MULTICALL2_ADDRESS "0x5ba1e12693dc8f9c48aad8770482f4739beed696"

#include <windows.h>
#include <bcrypt.h>
#include <winhttp.h>

#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#include "cuda_keccak.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "libcrypto.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "ws2_32.lib")

namespace
{
    constexpr wchar_t kDefaultRpcUrl[] = CONFIG_RPC_URL;
    constexpr int kDefaultWorkerCount = CONFIG_WORKER_THREADS;
    constexpr std::size_t kDefaultWalletCount = static_cast<std::size_t>(CONFIG_WALLETS_PER_BATCH);
    constexpr int kDefaultInflightRequests = CONFIG_INFLIGHT_REQUESTS;
    constexpr int kDefaultStatsIntervalSeconds = CONFIG_STATS_INTERVAL_SECONDS;
    constexpr int kDefaultRequestTimeoutMs = CONFIG_REQUEST_TIMEOUT_MS;
    constexpr char kFoundWalletsFile[] = CONFIG_FOUND_WALLETS_FILE;
    constexpr char kMulticall2ContractAddress[] = CONFIG_MULTICALL2_ADDRESS;
    constexpr char kUsdcContractAddress[] = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";
    constexpr char kUsdtContractAddress[] = "0xdAC17F958D2ee523a2206206994597C13D831ec7";
    constexpr char kBalanceOfSelector[] = "0x70a08231";
    constexpr char kMulticallAggregateSelector[] = "252dba42";
    constexpr char kZeroAddress[] = "0x0000000000000000000000000000000000000000";
    constexpr int kEthDecimals = 18;
    constexpr int kUsdcDecimals = 6;
    constexpr int kUsdtDecimals = 6;

    std::atomic<bool> gStopRequested{ false };
    std::mutex gPrintMutex;
    std::once_flag gCudaFallbackPrintOnce;

    using PrivateKey = std::array<std::uint8_t, 32>;
    using PublicKey = std::array<std::uint8_t, 64>;
    using Hash256 = std::array<std::uint8_t, 32>;
    using MicrosecondClock = std::chrono::steady_clock;
    constexpr std::size_t kPrivateKeyByteCount = 32;
    constexpr std::size_t kPublicKeyByteCount = 64;
    constexpr std::size_t kAddressByteCount = 20;
    constexpr std::size_t kAddressHexLength = 42;
    constexpr std::size_t kGpuHashMaxBatchKeys = 8192;
    constexpr auto kGpuHashBatchFlushDelay = std::chrono::microseconds(250);
#ifdef ETH_USE_CUDA
    std::atomic<std::uint64_t> gGpuDispatchBatches{ 0 };
    std::atomic<std::uint64_t> gGpuDispatchKeys{ 0 };
#endif

    static_assert(PrivateKey{}.size() == kPrivateKeyByteCount, "Unexpected PrivateKey size.");
    static_assert(PublicKey{}.size() == kPublicKeyByteCount, "Unexpected PublicKey size.");

    constexpr PrivateKey kSecp256k1Order = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
        0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
    };

    struct WalletEntry
    {
        int id;
        int ethRequestId;
        int usdcRequestId;
        int usdtRequestId;
        PrivateKey privateKey;
        std::string address;
    };

    struct RuntimeStats
    {
        std::atomic<std::uint64_t> totalWallets{ 0 };
        std::atomic<std::uint64_t> totalBatches{ 0 };
        std::atomic<std::uint64_t> totalErrors{ 0 };
        std::atomic<std::uint64_t> foundWallets{ 0 };
        std::atomic<std::uint64_t> timeGenWallets{ 0 };
        std::atomic<std::uint64_t> timeRandomPrivateKeys{ 0 };
        std::atomic<std::uint64_t> timePublicKeyDerivation{ 0 };
        std::atomic<std::uint64_t> timeAddressHash{ 0 };
        std::atomic<std::uint64_t> timeBuildRequest{ 0 };
        std::atomic<std::uint64_t> timeHttpRoundTrip{ 0 };
        std::atomic<std::uint64_t> timeParseBatch{ 0 };
        std::atomic<std::uint64_t> totalCudaHashBatches{ 0 };
        std::atomic<std::uint64_t> totalCpuHashBatches{ 0 };
        std::atomic<std::uint64_t> totalCudaFallbackBatches{ 0 };
    };

    struct WalletBuildMetrics
    {
        std::uint64_t randomFillUs = 0;
        std::uint64_t publicKeyDerivationUs = 0;
        std::uint64_t addressHashUs = 0;
        bool hashedAddresses = false;
        bool usedGpu = false;
        bool gpuFallback = false;
        std::string gpuError;
    };

    struct RpcUrlParts
    {
        std::wstring host;
        INTERNET_PORT port = INTERNET_DEFAULT_HTTP_PORT;
        std::wstring path = L"/";
        DWORD requestFlags = 0;
    };

    struct RuntimeConfig
    {
        int workerCount = kDefaultWorkerCount;
        std::size_t walletsPerBatch = kDefaultWalletCount;
        std::size_t multicallWalletsPerCall = kDefaultWalletCount;
        int inflightRequests = kDefaultInflightRequests;
        int statsIntervalSeconds = kDefaultStatsIntervalSeconds;
        int requestTimeoutMs = kDefaultRequestTimeoutMs;
    };

    enum class TokenRpcMode
    {
        DirectCalls,
        Multicall2Aggregate
    };

    struct TokenRpcSettings
    {
        TokenRpcMode mode = TokenRpcMode::DirectCalls;
        std::string multicallContractAddress = kMulticall2ContractAddress;
        std::size_t walletsPerMulticall = kDefaultWalletCount;
        std::string backendDescription = "Direct eth_call for each token";
    };

    bool IsHexCharacter(char ch)
    {
        return (ch >= '0' && ch <= '9') ||
            (ch >= 'a' && ch <= 'f') ||
            (ch >= 'A' && ch <= 'F');
    }

    std::string NormalizeAddressOrThrow(const std::string& value)
    {
        std::string out = value;
        if (out.size() >= 2 && out[0] == '0' && (out[1] == 'x' || out[1] == 'X')) {
            out = out.substr(2);
        }

        if (out.size() != 40) {
            throw std::runtime_error("Invalid address (length is not 40 hex chars): " + value);
        }

        for (char ch : out) {
            if (!IsHexCharacter(ch)) {
                throw std::runtime_error("Invalid address (non-hex character): " + value);
            }
        }

        return "0x" + out;
    }

    std::string Narrow(const std::wstring& value)
    {
        if (value.empty()) {
            return {};
        }

        const int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
        std::string out(size, '\0');
        WideCharToMultiByte(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), out.data(), size, nullptr, nullptr);
        return out;
    }

    int ReadEnvIntOrDefault(const char* name, int defaultValue)
    {
        char buffer[64] = {};
        const DWORD length = GetEnvironmentVariableA(name, buffer, static_cast<DWORD>(std::size(buffer)));
        if (length == 0 || length >= std::size(buffer)) {
            return defaultValue;
        }

        char* end = nullptr;
        const long parsed = std::strtol(buffer, &end, 10);
        if (end == buffer || (end != nullptr && *end != '\0')) {
            return defaultValue;
        }

        if (parsed < 1) {
            return defaultValue;
        }

        return static_cast<int>(parsed);
    }

    RuntimeConfig BuildRuntimeConfig()
    {
        RuntimeConfig config;
        config.workerCount = ReadEnvIntOrDefault("ETH_WORKER_THREADS", config.workerCount);
        config.walletsPerBatch = static_cast<std::size_t>(ReadEnvIntOrDefault("ETH_WALLETS_PER_BATCH", static_cast<int>(config.walletsPerBatch)));
        config.multicallWalletsPerCall = static_cast<std::size_t>(ReadEnvIntOrDefault("ETH_MULTICALL_WALLETS_PER_CALL", static_cast<int>(config.walletsPerBatch)));
        config.inflightRequests = ReadEnvIntOrDefault("ETH_INFLIGHT_REQUESTS", config.inflightRequests);
        config.statsIntervalSeconds = ReadEnvIntOrDefault("ETH_STATS_INTERVAL_SECONDS", config.statsIntervalSeconds);
        config.requestTimeoutMs = ReadEnvIntOrDefault("ETH_REQUEST_TIMEOUT_MS", config.requestTimeoutMs);

        config.workerCount = std::clamp(config.workerCount, 1, 256);
        config.walletsPerBatch = std::clamp<std::size_t>(config.walletsPerBatch, 1, 4096);
        config.multicallWalletsPerCall = std::clamp<std::size_t>(config.multicallWalletsPerCall, 1, 4096);
        config.inflightRequests = std::clamp(config.inflightRequests, 1, 16);
        config.statsIntervalSeconds = std::clamp(config.statsIntervalSeconds, 1, 3600);
        config.requestTimeoutMs = std::clamp(config.requestTimeoutMs, 1000, 300000);
        return config;
    }

    std::string BytesToHexLower(const std::uint8_t* data, std::size_t size)
    {
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out;
        out.resize(size * 2);

        for (std::size_t i = 0; i < size; ++i) {
            out[i * 2] = kHex[(data[i] >> 4) & 0x0F];
            out[i * 2 + 1] = kHex[data[i] & 0x0F];
        }

        return out;
    }

    std::string Strip0x(const std::string& value)
    {
        if (value.size() >= 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
            return value.substr(2);
        }
        return value;
    }

    std::string FormatHexAmount(const std::string& amountHex, int decimals)
    {
        if (decimals < 0) {
            throw std::runtime_error("Invalid decimal count.");
        }

        std::string hex = Strip0x(amountHex);
        if (hex.empty()) {
            hex = "0";
        }

        BIGNUM* value = nullptr;
        if (BN_hex2bn(&value, hex.c_str()) == 0 || value == nullptr) {
            throw std::runtime_error("Unable to convert hex balance to BIGNUM.");
        }

        char* decimalChars = BN_bn2dec(value);
        if (decimalChars == nullptr) {
            BN_free(value);
            throw std::runtime_error("Unable to convert balance to decimal string.");
        }

        std::string decimal(decimalChars);
        OPENSSL_free(decimalChars);
        BN_free(value);

        if (decimals == 0) {
            return decimal;
        }

        const std::size_t decimalsSize = static_cast<std::size_t>(decimals);
        if (decimal.size() <= decimalsSize) {
            decimal.insert(0, decimalsSize - decimal.size(), '0');
            decimal.insert(decimal.begin(), '0');
        }

        const std::size_t split = decimal.size() - decimalsSize;
        std::string integerPart = decimal.substr(0, split);
        std::string fractionalPart = decimal.substr(split);

        while (fractionalPart.size() > 1 && fractionalPart.back() == '0') {
            fractionalPart.pop_back();
        }

        return integerPart + "." + fractionalPart;
    }

    std::string FormatWeiHexAsEth(const std::string& weiHex)
    {
        return FormatHexAmount(weiHex, kEthDecimals);
    }

    std::string BuildBalanceOfData(const std::string& address)
    {
        const std::string normalizedAddress = Strip0x(address);
        if (normalizedAddress.size() != 40) {
            throw std::runtime_error("Invalid Ethereum address for balanceOf: " + address);
        }

        return std::string(kBalanceOfSelector) + std::string(64 - normalizedAddress.size(), '0') + normalizedAddress;
    }

    std::size_t AlignToAbiWord(std::size_t value)
    {
        return ((value + 31) / 32) * 32;
    }

    void AppendHexWord(std::string& out, std::uint64_t value)
    {
        char buffer[32] = {};
        const auto [ptr, ec] = std::to_chars(std::begin(buffer), std::end(buffer), value, 16);
        if (ec != std::errc{}) {
            throw std::runtime_error("Failed numeric ABI conversion.");
        }

        const std::size_t digitCount = static_cast<std::size_t>(ptr - buffer);
        out.append(64 - digitCount, '0');
        out.append(buffer, ptr);
    }

    void AppendPaddedHexWord(std::string& out, const std::string& hexValue)
    {
        const std::string normalizedHex = Strip0x(hexValue);
        if (normalizedHex.size() > 64) {
            throw std::runtime_error("Hex value is too long for one ABI word.");
        }

        out.append(64 - normalizedHex.size(), '0');
        out.append(normalizedHex);
    }

    void AppendAbiBytes(std::string& out, const std::string& hexValue)
    {
        const std::string normalizedHex = Strip0x(hexValue);
        if ((normalizedHex.size() % 2) != 0) {
            throw std::runtime_error("ABI hex payload has an odd number of characters.");
        }

        const std::size_t dataSizeBytes = normalizedHex.size() / 2;
        AppendHexWord(out, static_cast<std::uint64_t>(dataSizeBytes));
        out.append(normalizedHex);
        out.append((AlignToAbiWord(dataSizeBytes) * 2) - normalizedHex.size(), '0');
    }

    std::size_t GetMulticallTupleEncodedSizeBytes(const std::string& callData)
    {
        const std::size_t callDataSizeBytes = Strip0x(callData).size() / 2;
        return 32 + 32 + 32 + AlignToAbiWord(callDataSizeBytes);
    }

    void AppendMulticallTuple(std::string& out, const std::string& targetAddress, const std::string& callData)
    {
        AppendPaddedHexWord(out, targetAddress);
        AppendHexWord(out, 64);
        AppendAbiBytes(out, callData);
    }

    int HexNibbleOrThrow(char ch)
    {
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'f') {
            return 10 + (ch - 'a');
        }
        if (ch >= 'A' && ch <= 'F') {
            return 10 + (ch - 'A');
        }
        throw std::runtime_error(std::string("Invalid hex character: ") + ch);
    }

    std::vector<std::uint8_t> HexToBytesOrThrow(const std::string& value)
    {
        const std::string hex = Strip0x(value);
        if ((hex.size() % 2) != 0) {
            throw std::runtime_error("Hex string has an odd number of characters.");
        }

        std::vector<std::uint8_t> out(hex.size() / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const int hi = HexNibbleOrThrow(hex[i * 2]);
            const int lo = HexNibbleOrThrow(hex[(i * 2) + 1]);
            out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
        }
        return out;
    }

    std::uint64_t ReadAbiWordU64OrThrow(const std::vector<std::uint8_t>& data, std::size_t offset)
    {
        if ((offset + 32) > data.size()) {
            throw std::runtime_error("ABI offset out of bounds.");
        }

        for (std::size_t i = 0; i < 24; ++i) {
            if (data[offset + i] != 0) {
                throw std::runtime_error("ABI word is too large for the local parser.");
            }
        }

        std::uint64_t value = 0;
        for (std::size_t i = 24; i < 32; ++i) {
            value = (value << 8) | data[offset + i];
        }
        return value;
    }

    std::vector<std::string> DecodeMulticallAggregateResultsHex(const std::string& aggregateHex, std::size_t expectedCallCount)
    {
        const std::vector<std::uint8_t> decoded = HexToBytesOrThrow(aggregateHex);
        if (decoded.size() < 64) {
            throw std::runtime_error("Multicall response is too short.");
        }

        const std::size_t returnDataOffset = static_cast<std::size_t>(ReadAbiWordU64OrThrow(decoded, 32));
        if ((returnDataOffset + 32) > decoded.size()) {
            throw std::runtime_error("Invalid Multicall returnData offset.");
        }

        const std::size_t callCount = static_cast<std::size_t>(ReadAbiWordU64OrThrow(decoded, returnDataOffset));
        if (callCount != expectedCallCount) {
            throw std::runtime_error(
                "Unexpected Multicall result count: expected " + std::to_string(expectedCallCount) +
                ", received " + std::to_string(callCount) + ".");
        }

        const std::size_t arrayHeadBase = returnDataOffset + 32;
        std::vector<std::string> results(callCount);
        for (std::size_t i = 0; i < callCount; ++i) {
            const std::size_t elementOffset =
                static_cast<std::size_t>(ReadAbiWordU64OrThrow(decoded, arrayHeadBase + (i * 32)));
            const std::size_t elementBase = arrayHeadBase + elementOffset;
            const std::size_t elementLength = static_cast<std::size_t>(ReadAbiWordU64OrThrow(decoded, elementBase));
            const std::size_t dataStart = elementBase + 32;
            if ((dataStart + elementLength) > decoded.size()) {
                throw std::runtime_error("Multicall element is out of ABI response bounds.");
            }

            results[i] = "0x" + BytesToHexLower(decoded.data() + dataStart, elementLength);
        }

        return results;
    }

    std::uint64_t LoadLE64(const std::uint8_t* src)
    {
        std::uint64_t value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<std::uint64_t>(src[i]) << (8 * i);
        }
        return value;
    }

    void StoreLE64(std::uint8_t* dst, std::uint64_t value)
    {
        for (int i = 0; i < 8; ++i) {
            dst[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
        }
    }

    std::uint64_t RotL64(std::uint64_t value, int shift)
    {
        return (value << shift) | (value >> (64 - shift));
    }

    void KeccakF1600(std::uint64_t state[25])
    {
        static constexpr std::uint64_t kRoundConstants[24] = {
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

        static constexpr int kRotationOffsets[24] = {
            1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
           27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
        };

        static constexpr int kPiLane[24] = {
            10,  7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
            15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
        };

        for (int round = 0; round < 24; ++round) {
            std::uint64_t c[5] = {};
            for (int x = 0; x < 5; ++x) {
                c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }

            std::uint64_t d[5] = {};
            for (int x = 0; x < 5; ++x) {
                d[x] = c[(x + 4) % 5] ^ RotL64(c[(x + 1) % 5], 1);
            }

            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 25; y += 5) {
                    state[x + y] ^= d[x];
                }
            }

            std::uint64_t current = state[1];
            for (int i = 0; i < 24; ++i) {
                const int lane = kPiLane[i];
                const std::uint64_t tmp = state[lane];
                state[lane] = RotL64(current, kRotationOffsets[i]);
                current = tmp;
            }

            for (int y = 0; y < 25; y += 5) {
                std::uint64_t row[5] = {};
                for (int x = 0; x < 5; ++x) {
                    row[x] = state[y + x];
                }
                for (int x = 0; x < 5; ++x) {
                    state[y + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
                }
            }

            state[0] ^= kRoundConstants[round];
        }
    }

    Hash256 Keccak256(const std::uint8_t* data, std::size_t size)
    {
        constexpr std::size_t kRate = 136;
        std::uint64_t state[25] = {};
        std::size_t offset = 0;

        while (size >= kRate) {
            for (std::size_t i = 0; i < kRate / 8; ++i) {
                state[i] ^= LoadLE64(data + offset + i * 8);
            }
            KeccakF1600(state);
            offset += kRate;
            size -= kRate;
        }

        std::array<std::uint8_t, kRate> block = {};
        for (std::size_t i = 0; i < size; ++i) {
            block[i] = data[offset + i];
        }
        block[size] = 0x01;
        block[kRate - 1] |= 0x80;

        for (std::size_t i = 0; i < kRate / 8; ++i) {
            state[i] ^= LoadLE64(block.data() + i * 8);
        }
        KeccakF1600(state);

        Hash256 hash = {};
        for (std::size_t i = 0; i < hash.size() / 8; ++i) {
            StoreLE64(hash.data() + i * 8, state[i]);
        }
        return hash;
    }

    class ThreadLocalOpenSSLContext
    {
    public:
        ThreadLocalOpenSSLContext() {
            ecKey_ = EC_KEY_new_by_curve_name(NID_secp256k1);
            ctx_ = BN_CTX_new();
            if (ecKey_ == nullptr || ctx_ == nullptr) {
                EC_KEY_free(ecKey_);
                BN_CTX_free(ctx_);
                ecKey_ = nullptr;
                ctx_ = nullptr;
                throw std::runtime_error("OpenSSL allocation failed in thread-local context.");
            }

            group_ = EC_KEY_get0_group(ecKey_);
            (void)EC_KEY_precompute_mult(ecKey_, ctx_);
            privateScalar_ = BN_new();
            publicPoint_ = (group_ != nullptr) ? EC_POINT_new(group_) : nullptr;
            if (group_ == nullptr || privateScalar_ == nullptr || publicPoint_ == nullptr) {
                EC_POINT_free(publicPoint_);
                BN_free(privateScalar_);
                EC_KEY_free(ecKey_);
                BN_CTX_free(ctx_);
                publicPoint_ = nullptr;
                privateScalar_ = nullptr;
                ecKey_ = nullptr;
                ctx_ = nullptr;
                throw std::runtime_error("OpenSSL initialization failed in thread-local context.");
            }
        }

        ~ThreadLocalOpenSSLContext() {
            EC_POINT_free(publicPoint_);
            BN_free(privateScalar_);
            EC_KEY_free(ecKey_);
            BN_CTX_free(ctx_);
        }

        const EC_GROUP* group() const { return group_; }
        BN_CTX* ctx() const { return ctx_; }
        BIGNUM* privateScalar() const { return privateScalar_; }
        EC_POINT* publicPoint() const { return publicPoint_; }

        ThreadLocalOpenSSLContext(const ThreadLocalOpenSSLContext&) = delete;
        ThreadLocalOpenSSLContext& operator=(const ThreadLocalOpenSSLContext&) = delete;

    private:
        EC_KEY* ecKey_ = nullptr;
        const EC_GROUP* group_ = nullptr;
        BN_CTX* ctx_ = nullptr;
        BIGNUM* privateScalar_ = nullptr;
        EC_POINT* publicPoint_ = nullptr;
    };

    thread_local std::unique_ptr<ThreadLocalOpenSSLContext> gThreadLocalContext;

    ThreadLocalOpenSSLContext& GetThreadLocalContext() {
        if (!gThreadLocalContext) {
            gThreadLocalContext = std::make_unique<ThreadLocalOpenSSLContext>();
        }
        return *gThreadLocalContext;
    }

    PublicKey PrivateKeyToPublicKey(const PrivateKey& privateKey)
    {
        auto& tlContext = GetThreadLocalContext();
        BN_CTX* ctx = tlContext.ctx();
        const EC_GROUP* group = tlContext.group();
        BIGNUM* priv = BN_bin2bn(privateKey.data(), static_cast<int>(privateKey.size()), tlContext.privateScalar());
        EC_POINT* pub = tlContext.publicPoint();

        if (group == nullptr || priv == nullptr || pub == nullptr) {
            throw std::runtime_error("OpenSSL allocation failed.");
        }

        if (EC_POINT_mul(group, pub, priv, nullptr, nullptr, ctx) != 1) {
            throw std::runtime_error("Public key derivation failed.");
        }

        std::array<unsigned char, 65> publicKey = {};
        const size_t publicKeySize = EC_POINT_point2oct(
            group,
            pub,
            POINT_CONVERSION_UNCOMPRESSED,
            publicKey.data(),
            publicKey.size(),
            ctx
        );

        if (publicKeySize != 65 || publicKey[0] != 0x04) {
            throw std::runtime_error("Invalid public key format.");
        }

        PublicKey publicKeyNoPrefix = {};
        std::copy(publicKey.begin() + 1, publicKey.end(), publicKeyNoPrefix.begin());
        return publicKeyNoPrefix;
    }

    void SetAddressFromBytes(std::string& out, const std::uint8_t* addressBytes)
    {
        static constexpr char kHex[] = "0123456789abcdef";
        out.resize(kAddressHexLength);
        out[0] = '0';
        out[1] = 'x';

        for (std::size_t i = 0; i < kAddressByteCount; ++i) {
            const std::uint8_t byte = addressBytes[i];
            out[2 + (i * 2)] = kHex[(byte >> 4) & 0x0F];
            out[3 + (i * 2)] = kHex[byte & 0x0F];
        }
    }

    void SetAddressFromPublicKeyCpu(std::string& out, const PublicKey& publicKey)
    {
        const Hash256 hash = Keccak256(publicKey.data(), publicKey.size());
        SetAddressFromBytes(out, hash.data() + 12);
    }

    bool IsZeroPrivateKey(const PrivateKey& key)
    {
        std::uint8_t accum = 0;
        for (std::uint8_t b : key) {
            accum |= b;
        }
        return accum == 0;
    }

    int CompareBigEndian(const PrivateKey& lhs, const PrivateKey& rhs)
    {
        const int cmp = std::memcmp(lhs.data(), rhs.data(), lhs.size());
        return (cmp < 0) ? -1 : ((cmp > 0) ? 1 : 0);
    }

    bool IsValidSecp256k1PrivateKey(const PrivateKey& key)
    {
        return !IsZeroPrivateKey(key) && CompareBigEndian(key, kSecp256k1Order) < 0;
    }

    void FillRandomBytesOrThrow(std::uint8_t* dst, std::size_t size)
    {
        const NTSTATUS status = BCryptGenRandom(
            nullptr,
            dst,
            static_cast<ULONG>(size),
            BCRYPT_USE_SYSTEM_PREFERRED_RNG
        );
        if (status < 0) {
            throw std::runtime_error("BCryptGenRandom failed while generating a private key.");
        }
    }

    struct ThreadLocalWalletBuildScratch
    {
        std::vector<PrivateKey> randomPrivateKeys;
        std::vector<PublicKey> publicKeys;
    };

    thread_local ThreadLocalWalletBuildScratch gThreadLocalWalletBuildScratch;

    ThreadLocalWalletBuildScratch& GetThreadLocalWalletBuildScratch(std::size_t walletCount)
    {
        ThreadLocalWalletBuildScratch& scratch = gThreadLocalWalletBuildScratch;
        if (scratch.randomPrivateKeys.size() != walletCount) {
            scratch.randomPrivateKeys.resize(walletCount);
        }
        if (scratch.publicKeys.size() != walletCount) {
            scratch.publicKeys.resize(walletCount);
        }
        return scratch;
    }

    void AppendInt(std::string& out, int value)
    {
        char buffer[16] = {};
        const auto [ptr, ec] = std::to_chars(std::begin(buffer), std::end(buffer), value);
        if (ec != std::errc{}) {
            throw std::runtime_error("Numeric-to-string conversion failed.");
        }
        out.append(buffer, ptr);
    }

    using BatchRpcResults = std::vector<std::string>;

    const std::string& GetBatchResultHexOrZero(const BatchRpcResults& balances, int requestId)
    {
        static const std::string kZeroHexAmount = "0x0";
        const std::size_t index = static_cast<std::size_t>(requestId);
        if (index < balances.size() && !balances[index].empty()) {
            return balances[index];
        }
        return kZeroHexAmount;
    }

#ifdef ETH_USE_CUDA
    struct GpuHashResult
    {
        bool success = false;
        std::vector<std::uint8_t> addressBytes;
        std::string error;
    };

    struct GpuHashRequest
    {
        // The caller waits synchronously on the future, so this pointer stays valid
        // until the GPU batching thread has copied the input keys into its own buffer.
        const PrivateKey* privateKeys = nullptr;
        std::size_t keyCount = 0;
        std::promise<GpuHashResult> promise;
    };

    class GpuAddressHashService
    {
    public:
        GpuHashResult Hash(const std::vector<PrivateKey>& privateKeys)
        {
            if (privateKeys.empty()) {
                return { true, {}, {} };
            }

            std::future<GpuHashResult> future;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!fatalError_.empty()) {
                    return { false, {}, fatalError_ };
                }
                if (stopRequested_) {
                    return { false, {}, "CUDA service is shutting down." };
                }

                GpuHashRequest request;
                request.privateKeys = privateKeys.data();
                request.keyCount = privateKeys.size();
                future = request.promise.get_future();
                queue_.push_back(std::move(request));
            }

            signal_.notify_one();
            return future.get();
        }

        void Shutdown()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopRequested_) {
                    return;
                }
                stopRequested_ = true;
            }

            signal_.notify_one();
            if (worker_.joinable()) {
                worker_.join();
            }
        }

        ~GpuAddressHashService()
        {
            Shutdown();
        }

    private:
        GpuAddressHashService()
            : worker_(&GpuAddressHashService::Run, this)
        {
        }

        friend GpuAddressHashService& GetGpuAddressHashService();

        static GpuHashResult MakeFailure(const std::string& error)
        {
            GpuHashResult result;
            result.success = false;
            result.error = error;
            return result;
        }

        void ResolveBatchFailure(std::vector<GpuHashRequest>& requests, const std::string& error)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (fatalError_.empty()) {
                    fatalError_ = error;
                }
            }

            for (auto& request : requests) {
                request.promise.set_value(MakeFailure(error));
            }
        }

        void Run()
        {
            std::vector<GpuHashRequest> batchedRequests;
            std::vector<std::uint8_t> batchedPrivateKeys;
            std::vector<std::uint8_t> batchedAddresses;

            for (;;) {
                batchedRequests.clear();
                std::size_t totalKeys = 0;

                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    signal_.wait(lock, [&]() {
                        return stopRequested_ || !queue_.empty();
                        });

                    if (stopRequested_ && queue_.empty()) {
                        break;
                    }

                    const auto deadline = MicrosecondClock::now() + kGpuHashBatchFlushDelay;
                    while (totalKeys < kGpuHashMaxBatchKeys) {
                        while (!queue_.empty() && totalKeys < kGpuHashMaxBatchKeys) {
                            GpuHashRequest request = std::move(queue_.front());
                            queue_.pop_front();
                            totalKeys += request.keyCount;
                            batchedRequests.push_back(std::move(request));
                        }

                        if (totalKeys >= kGpuHashMaxBatchKeys || stopRequested_) {
                            break;
                        }
                        if (MicrosecondClock::now() >= deadline) {
                            break;
                        }
                        if (queue_.empty() && signal_.wait_until(lock, deadline) == std::cv_status::timeout) {
                            break;
                        }
                    }
                }

                if (batchedRequests.empty()) {
                    continue;
                }

                try {
                    batchedPrivateKeys.resize(totalKeys * kPrivateKeyByteCount);
                    batchedAddresses.resize(totalKeys * kAddressByteCount);

                    std::size_t keyOffset = 0;
                    for (const auto& request : batchedRequests) {
                        const std::size_t privateKeyBytes = request.keyCount * kPrivateKeyByteCount;
                        std::memcpy(
                            batchedPrivateKeys.data() + (keyOffset * kPrivateKeyByteCount),
                            reinterpret_cast<const std::uint8_t*>(request.privateKeys),
                            privateKeyBytes);
                        keyOffset += request.keyCount;
                    }

                    std::string error;
                    if (!eth_cuda::ComputeEthereumAddressesFromPrivateKeysCuda(
                        batchedPrivateKeys.data(),
                        totalKeys,
                        batchedAddresses.data(),
                        error)) {
                        if (error.empty()) {
                            error = "CUDA service returned an unknown error.";
                        }
                        ResolveBatchFailure(batchedRequests, error);
                        continue;
                    }

                    gGpuDispatchBatches.fetch_add(1, std::memory_order_relaxed);
                    gGpuDispatchKeys.fetch_add(totalKeys, std::memory_order_relaxed);

                    std::size_t addressOffset = 0;
                    for (auto& request : batchedRequests) {
                        GpuHashResult result;
                        result.success = true;
                        result.addressBytes.resize(request.keyCount * kAddressByteCount);
                        std::memcpy(
                            result.addressBytes.data(),
                            batchedAddresses.data() + (addressOffset * kAddressByteCount),
                            result.addressBytes.size());
                        addressOffset += request.keyCount;
                        request.promise.set_value(std::move(result));
                    }
                }
                catch (const std::exception& ex) {
                    ResolveBatchFailure(batchedRequests, std::string("CUDA batching service failed: ") + ex.what());
                }
            }
        }

        std::mutex mutex_;
        std::condition_variable signal_;
        std::deque<GpuHashRequest> queue_;
        std::thread worker_;
        bool stopRequested_ = false;
        std::string fatalError_;
    };

    GpuAddressHashService& GetGpuAddressHashService()
    {
        static GpuAddressHashService service;
        return service;
    }

    GpuHashResult HashAddressesOnGpu(const std::vector<PrivateKey>& privateKeys)
    {
        try {
            return GetGpuAddressHashService().Hash(privateKeys);
        }
        catch (const std::exception& ex) {
            GpuHashResult result;
            result.success = false;
            result.error = std::string("Unable to use CUDA batching: ") + ex.what();
            return result;
        }
    }

    bool ValidateCudaAddressPipeline(std::string& error)
    {
        constexpr std::size_t kValidationKeyCount = 16;
        std::array<PrivateKey, kValidationKeyCount> validationKeys = {};
        for (std::size_t i = 0; i < validationKeys.size(); ++i) {
            validationKeys[i].fill(0);
            validationKeys[i].back() = static_cast<std::uint8_t>(i + 1);
        }

        for (std::size_t i = 8; i < validationKeys.size(); ++i) {
            do {
                FillRandomBytesOrThrow(validationKeys[i].data(), validationKeys[i].size());
            } while (!IsValidSecp256k1PrivateKey(validationKeys[i]));
        }

        std::vector<std::uint8_t> gpuAddressBytes(validationKeys.size() * kAddressByteCount);
        if (!eth_cuda::ComputeEthereumAddressesFromPrivateKeysCuda(
            reinterpret_cast<const std::uint8_t*>(validationKeys.data()),
            validationKeys.size(),
            gpuAddressBytes.data(),
            error)) {
            return false;
        }

        for (std::size_t i = 0; i < validationKeys.size(); ++i) {
            const PublicKey publicKey = PrivateKeyToPublicKey(validationKeys[i]);
            std::string cpuAddress;
            std::string gpuAddress;
            SetAddressFromPublicKeyCpu(cpuAddress, publicKey);
            SetAddressFromBytes(gpuAddress, gpuAddressBytes.data() + (i * kAddressByteCount));

            if (cpuAddress != gpuAddress) {
                error =
                    "CUDA secp256k1 validation failed on sample " + std::to_string(i) +
                    " (cpu=" + cpuAddress + ", gpu=" + gpuAddress + ").";
                return false;
            }
        }

        return true;
    }
#endif

    std::vector<WalletEntry> BuildRandomWallets(
        std::size_t walletCount,
        const std::string& forcedAddress,
        bool enableCudaHashing,
        WalletBuildMetrics& metrics)
    {
        std::vector<WalletEntry> wallets(walletCount);
        const bool useForcedAddress = !forcedAddress.empty();
        ThreadLocalWalletBuildScratch& scratch = GetThreadLocalWalletBuildScratch(walletCount);

        const auto t_random_start = MicrosecondClock::now();
        FillRandomBytesOrThrow(
            reinterpret_cast<std::uint8_t*>(scratch.randomPrivateKeys.data()),
            scratch.randomPrivateKeys.size() * sizeof(PrivateKey));

        for (std::size_t i = 0; i < walletCount; ++i) {
            PrivateKey& key = scratch.randomPrivateKeys[i];
            WalletEntry& entry = wallets[i];

            do {
                if (IsValidSecp256k1PrivateKey(key)) {
                    break;
                }
                FillRandomBytesOrThrow(key.data(), key.size());
            } while (!IsValidSecp256k1PrivateKey(key));

            entry.privateKey = key;

            if (useForcedAddress) {
                entry.address = forcedAddress;
            }
        }
        const auto t_random_end = MicrosecondClock::now();
        metrics.randomFillUs = std::chrono::duration_cast<std::chrono::microseconds>(t_random_end - t_random_start).count();

        if (useForcedAddress) {
            return wallets;
        }

        metrics.hashedAddresses = true;
        const auto t_hash_start = MicrosecondClock::now();

#ifdef ETH_USE_CUDA
        if (enableCudaHashing) {
            GpuHashResult gpuResult = HashAddressesOnGpu(scratch.randomPrivateKeys);
            if (gpuResult.success && gpuResult.addressBytes.size() == (wallets.size() * kAddressByteCount)) {
                metrics.usedGpu = true;
                for (std::size_t i = 0; i < wallets.size(); ++i) {
                    SetAddressFromBytes(wallets[i].address, gpuResult.addressBytes.data() + (i * kAddressByteCount));
                }
            } else {
                metrics.gpuFallback = true;
                metrics.gpuError = gpuResult.error.empty()
                    ? "CUDA batching returned output with unexpected size."
                    : std::move(gpuResult.error);

                std::vector<PublicKey>& publicKeys = scratch.publicKeys;
                publicKeys.resize(wallets.size());

                const auto t_public_key_start = MicrosecondClock::now();
                for (std::size_t i = 0; i < wallets.size(); ++i) {
                    publicKeys[i] = PrivateKeyToPublicKey(wallets[i].privateKey);
                }
                const auto t_public_key_end = MicrosecondClock::now();
                metrics.publicKeyDerivationUs = std::chrono::duration_cast<std::chrono::microseconds>(t_public_key_end - t_public_key_start).count();

                for (std::size_t i = 0; i < wallets.size(); ++i) {
                    SetAddressFromPublicKeyCpu(wallets[i].address, publicKeys[i]);
                }
            }
        } else
#endif
        {
            std::vector<PublicKey>& publicKeys = scratch.publicKeys;
            publicKeys.resize(wallets.size());

            const auto t_public_key_start = MicrosecondClock::now();
            for (std::size_t i = 0; i < wallets.size(); ++i) {
                publicKeys[i] = PrivateKeyToPublicKey(wallets[i].privateKey);
            }
            const auto t_public_key_end = MicrosecondClock::now();
            metrics.publicKeyDerivationUs = std::chrono::duration_cast<std::chrono::microseconds>(t_public_key_end - t_public_key_start).count();

            for (std::size_t i = 0; i < wallets.size(); ++i) {
                SetAddressFromPublicKeyCpu(wallets[i].address, publicKeys[i]);
            }
        }

        const auto t_hash_end = MicrosecondClock::now();
        metrics.addressHashUs = std::chrono::duration_cast<std::chrono::microseconds>(t_hash_end - t_hash_start).count();

        return wallets;
    }

    struct MulticallChunkRequest
    {
        int requestId = 0;
        std::size_t walletStartIndex = 0;
        std::size_t walletCount = 0;
    };

    struct BatchRequestBuildResult
    {
        std::string json;
        std::size_t requestCount = 0;
        std::vector<MulticallChunkRequest> multicallChunks;
    };

    void AppendEthGetBalanceJson(std::string& out, bool& firstItem, int requestId, const std::string& address)
    {
        if (!firstItem) {
            out.push_back(',');
        }
        firstItem = false;

        out.append("{\"jsonrpc\":\"2.0\",\"id\":");
        AppendInt(out, requestId);
        out.append(",\"method\":\"eth_getBalance\",\"params\":[\"");
        out.append(address);
        out.append("\",\"latest\"]}");
    }

    void AppendEthCallJson(
        std::string& out,
        bool& firstItem,
        int requestId,
        const std::string& contractAddress,
        const std::string& data)
    {
        if (!firstItem) {
            out.push_back(',');
        }
        firstItem = false;

        out.append("{\"jsonrpc\":\"2.0\",\"id\":");
        AppendInt(out, requestId);
        out.append(",\"method\":\"eth_call\",\"params\":[{\"to\":\"");
        out.append(contractAddress);
        out.append("\",\"data\":\"");
        out.append(data);
        out.append("\"},\"latest\"]}");
    }

    std::string BuildMulticallAggregateData(const std::vector<WalletEntry>& wallets, std::size_t startIndex, std::size_t walletCount)
    {
        if (walletCount == 0 || (startIndex + walletCount) > wallets.size()) {
            throw std::runtime_error("Invalid wallet range for BuildMulticallAggregateData.");
        }

        const std::size_t callCount = walletCount * 2;
        std::vector<std::string> balanceOfData(walletCount);
        for (std::size_t i = 0; i < walletCount; ++i) {
            balanceOfData[i] = BuildBalanceOfData(wallets[startIndex + i].address);
        }

        std::vector<std::size_t> tupleSizes(callCount);
        for (std::size_t i = 0; i < walletCount; ++i) {
            const std::size_t tupleSize = GetMulticallTupleEncodedSizeBytes(balanceOfData[i]);
            tupleSizes[i * 2] = tupleSize;
            tupleSizes[(i * 2) + 1] = tupleSize;
        }

        std::string out;
        out.reserve(2 + 8 + 64 + 64 + (callCount * 64) + (walletCount * 2 * 320));
        out = "0x";
        out.append(kMulticallAggregateSelector);
        AppendHexWord(out, 32);
        AppendHexWord(out, static_cast<std::uint64_t>(callCount));

        std::size_t runningOffset = callCount * 32;
        for (std::size_t i = 0; i < callCount; ++i) {
            AppendHexWord(out, static_cast<std::uint64_t>(runningOffset));
            runningOffset += tupleSizes[i];
        }

        for (std::size_t i = 0; i < walletCount; ++i) {
            AppendMulticallTuple(out, kUsdcContractAddress, balanceOfData[i]);
            AppendMulticallTuple(out, kUsdtContractAddress, balanceOfData[i]);
        }

        return out;
    }

    BatchRequestBuildResult BuildBatchRequest(std::vector<WalletEntry>& wallets, const TokenRpcSettings& tokenRpcSettings)
    {
        BatchRequestBuildResult result;
        result.multicallChunks.reserve(
            (tokenRpcSettings.mode == TokenRpcMode::Multicall2Aggregate)
                ? ((wallets.size() + tokenRpcSettings.walletsPerMulticall - 1) / tokenRpcSettings.walletsPerMulticall)
                : 0);

        result.json.reserve(
            (tokenRpcSettings.mode == TokenRpcMode::Multicall2Aggregate)
                ? ((wallets.size() * 140) + (result.multicallChunks.capacity() * 1200) + 2)
                : ((wallets.size() * 430) + 2));
        result.json.push_back('[');

        int nextRequestId = 1;
        bool firstItem = true;
        for (WalletEntry& wallet : wallets) {
            wallet.ethRequestId = nextRequestId++;
            wallet.usdcRequestId = 0;
            wallet.usdtRequestId = 0;
            AppendEthGetBalanceJson(result.json, firstItem, wallet.ethRequestId, wallet.address);
        }

        if (tokenRpcSettings.mode == TokenRpcMode::Multicall2Aggregate) {
            const std::size_t walletsPerChunk = std::max<std::size_t>(1, tokenRpcSettings.walletsPerMulticall);
            for (std::size_t startIndex = 0; startIndex < wallets.size(); startIndex += walletsPerChunk) {
                const std::size_t chunkWalletCount = std::min(walletsPerChunk, wallets.size() - startIndex);
                const int requestId = nextRequestId++;
                result.multicallChunks.push_back({ requestId, startIndex, chunkWalletCount });
                AppendEthCallJson(
                    result.json,
                    firstItem,
                    requestId,
                    tokenRpcSettings.multicallContractAddress,
                    BuildMulticallAggregateData(wallets, startIndex, chunkWalletCount));
            }
        } else {
            for (WalletEntry& wallet : wallets) {
                const std::string balanceOfData = BuildBalanceOfData(wallet.address);
                wallet.usdcRequestId = nextRequestId++;
                wallet.usdtRequestId = nextRequestId++;
                AppendEthCallJson(result.json, firstItem, wallet.usdcRequestId, kUsdcContractAddress, balanceOfData);
                AppendEthCallJson(result.json, firstItem, wallet.usdtRequestId, kUsdtContractAddress, balanceOfData);
            }
        }

        result.json.push_back(']');
        result.requestCount = static_cast<std::size_t>(nextRequestId - 1);
        return result;
    }

    bool IsZeroHexAmount(const std::string& value)
    {
        const std::string hex = Strip0x(value);
        if (hex.empty()) {
            return true;
        }

        for (char ch : hex) {
            if (ch != '0') {
                return false;
            }
        }
        return true;
    }

    RpcUrlParts ParseRpcUrl(const std::wstring& url)
    {
        URL_COMPONENTS components = {};
        components.dwStructSize = sizeof(components);

        wchar_t hostName[256] = {};
        wchar_t urlPath[2048] = {};
        wchar_t extraInfo[2048] = {};

        components.lpszHostName = hostName;
        components.dwHostNameLength = static_cast<DWORD>(sizeof(hostName) / sizeof(hostName[0]));
        components.lpszUrlPath = urlPath;
        components.dwUrlPathLength = static_cast<DWORD>(sizeof(urlPath) / sizeof(urlPath[0]));
        components.lpszExtraInfo = extraInfo;
        components.dwExtraInfoLength = static_cast<DWORD>(sizeof(extraInfo) / sizeof(extraInfo[0]));

        if (!WinHttpCrackUrl(url.c_str(), static_cast<DWORD>(url.size()), 0, &components)) {
            throw std::runtime_error("WinHttpCrackUrl failed.");
        }

        RpcUrlParts parts;
        parts.host.assign(components.lpszHostName, components.dwHostNameLength);
        parts.port = components.nPort;
        parts.path =
            std::wstring(components.lpszUrlPath, components.dwUrlPathLength) +
            std::wstring(components.lpszExtraInfo, components.dwExtraInfoLength);
        if (parts.path.empty()) {
            parts.path = L"/";
        }
        parts.requestFlags = (components.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
        return parts;
    }

    class WinHttpPersistentClient
    {
    public:
        explicit WinHttpPersistentClient(const std::wstring& rpcUrl, int requestTimeoutMs)
            : parts_(ParseRpcUrl(rpcUrl))
        {
            session_ = WinHttpOpen(
                L"EthBalanceBatch/2.0",
                WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                WINHTTP_NO_PROXY_NAME,
                WINHTTP_NO_PROXY_BYPASS,
                0
            );
            if (session_ == nullptr) {
                throw std::runtime_error("WinHttpOpen failed.");
            }

            if (!WinHttpSetTimeouts(session_, requestTimeoutMs, requestTimeoutMs, requestTimeoutMs, requestTimeoutMs)) {
                WinHttpCloseHandle(session_);
                session_ = nullptr;
                throw std::runtime_error("WinHttpSetTimeouts failed.");
            }

            connect_ = WinHttpConnect(session_, parts_.host.c_str(), parts_.port, 0);
            if (connect_ == nullptr) {
                WinHttpCloseHandle(session_);
                session_ = nullptr;
                throw std::runtime_error("WinHttpConnect failed.");
            }
        }

        ~WinHttpPersistentClient()
        {
            if (connect_ != nullptr) {
                WinHttpCloseHandle(connect_);
                connect_ = nullptr;
            }
            if (session_ != nullptr) {
                WinHttpCloseHandle(session_);
                session_ = nullptr;
            }
        }

        std::string PostJson(const std::string& jsonBody) const
        {
            HINTERNET request = WinHttpOpenRequest(
                connect_,
                L"POST",
                parts_.path.c_str(),
                nullptr,
                WINHTTP_NO_REFERER,
                WINHTTP_DEFAULT_ACCEPT_TYPES,
                parts_.requestFlags
            );
            if (request == nullptr) {
                throw std::runtime_error("WinHttpOpenRequest failed.");
            }

            const auto closeRequest = [&request]() {
                if (request != nullptr) {
                    WinHttpCloseHandle(request);
                    request = nullptr;
                }
                };

            const wchar_t* headers = L"Content-Type: application/json\r\nConnection: keep-alive\r\n";
            void* requestBody = jsonBody.empty() ? nullptr : const_cast<char*>(jsonBody.c_str());
            if (!WinHttpSendRequest(
                request,
                headers,
                static_cast<DWORD>(-1L),
                requestBody,
                static_cast<DWORD>(jsonBody.size()),
                static_cast<DWORD>(jsonBody.size()),
                0) ||
                !WinHttpReceiveResponse(request, nullptr)) {
                const DWORD errorCode = GetLastError();
                closeRequest();
                throw std::runtime_error("HTTP send/receive failed (WinHTTP error " + std::to_string(errorCode) + ").");
            }

            DWORD statusCode = 0;
            DWORD statusCodeSize = sizeof(statusCode);
            if (!WinHttpQueryHeaders(
                request,
                WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                WINHTTP_HEADER_NAME_BY_INDEX,
                &statusCode,
                &statusCodeSize,
                WINHTTP_NO_HEADER_INDEX)) {
                closeRequest();
                throw std::runtime_error("WinHttpQueryHeaders failed.");
            }

            std::string response;
            DWORD available = 0;
            do {
                available = 0;
                if (!WinHttpQueryDataAvailable(request, &available)) {
                    closeRequest();
                    throw std::runtime_error("WinHttpQueryDataAvailable failed.");
                }

                if (available == 0) {
                    break;
                }

                const std::size_t previousSize = response.size();
                response.resize(previousSize + available);
                DWORD downloaded = 0;
                char* writePtr = response.data() + previousSize;

                if (!WinHttpReadData(request, writePtr, available, &downloaded)) {
                    closeRequest();
                    throw std::runtime_error("WinHttpReadData failed.");
                }

                response.resize(previousSize + downloaded);
            } while (available > 0);

            closeRequest();
            if (statusCode < 200 || statusCode >= 300) {
                throw std::runtime_error("RPC server responded with HTTP " + std::to_string(statusCode) + ": " + response);
            }

            return response;
        }

        WinHttpPersistentClient(const WinHttpPersistentClient&) = delete;
        WinHttpPersistentClient& operator=(const WinHttpPersistentClient&) = delete;

    private:
        RpcUrlParts parts_;
        HINTERNET session_ = nullptr;
        HINTERNET connect_ = nullptr;
    };

    BatchRpcResults ParseRpcResultsById(const std::string& response, std::size_t requestCount)
    {
        BatchRpcResults balances(requestCount + 1);
        bool foundAny = false;

        const size_t len = response.size();
        size_t pos = 0;

        while (pos < len) {
            // Find next "id"
            const size_t idPos = response.find("\"id\":", pos);
            if (idPos == std::string::npos) {
                break;
            }

            // Parse id value
            size_t idStart = idPos + 5; // length of "\"id\":"
            while (idStart < len && (response[idStart] == ' ' || response[idStart] == '\t')) {
                ++idStart;
            }

            if (idStart >= len || !std::isdigit(response[idStart])) {
                pos = idStart;
                continue;
            }

            int id = 0;
            size_t idEnd = idStart;
            while (idEnd < len && std::isdigit(response[idEnd])) {
                id = id * 10 + (response[idEnd] - '0');
                ++idEnd;
            }

            // Find next "result"
            const size_t resultPos = response.find("\"result\":", idEnd);
            if (resultPos == std::string::npos) {
                pos = idEnd;
                continue;
            }

            // Skip to opening quote
            size_t valueStart = resultPos + 9; // length of "\"result\":"
            while (valueStart < len && (response[valueStart] == ' ' || response[valueStart] == '\t')) {
                ++valueStart;
            }

            if (valueStart >= len || response[valueStart] != '"') {
                pos = valueStart;
                continue;
            }

            // Extract hex value between quotes
            ++valueStart; // skip opening quote
            size_t valueEnd = valueStart;
            while (valueEnd < len && response[valueEnd] != '"') {
                ++valueEnd;
            }

            if (valueEnd < len) {
                if (id > 0) {
                    const std::size_t index = static_cast<std::size_t>(id);
                    if (index < balances.size()) {
                        balances[index] = response.substr(valueStart, valueEnd - valueStart);
                        foundAny = true;
                    }
                }
                pos = valueEnd + 1;
            } else {
                pos = valueStart;
            }
        }

        if (!foundAny) {
            throw std::runtime_error("No RPC result found in batch response. Raw response: " + response);
        }

        return balances;
    }

    std::string GetRequiredRpcResultHex(const std::string& response)
    {
        const BatchRpcResults results = ParseRpcResultsById(response, 1);
        if (results.size() <= 1 || results[1].empty()) {
            throw std::runtime_error("Missing RPC result in response: " + response);
        }
        return results[1];
    }

    bool ProbeMulticall2Support(
        const std::wstring& rpcUrl,
        int requestTimeoutMs,
        const TokenRpcSettings& tokenRpcSettings,
        std::string& error)
    {
        try {
            WinHttpPersistentClient client(rpcUrl, requestTimeoutMs);

            std::string getCodeRequest =
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_getCode\",\"params\":[\"" +
                tokenRpcSettings.multicallContractAddress +
                "\",\"latest\"]}";
            const std::string codeHex = GetRequiredRpcResultHex(client.PostJson(getCodeRequest));
            if (codeHex.empty() || codeHex == "0x") {
                error =
                    "eth_getCode found no bytecode at " +
                    tokenRpcSettings.multicallContractAddress + ".";
                return false;
            }

            std::vector<WalletEntry> probeWallets(1);
            probeWallets[0].address = kZeroAddress;
            const std::string probeData = BuildMulticallAggregateData(probeWallets, 0, 1);

            std::string probeRequest =
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_call\",\"params\":[{\"to\":\"" +
                tokenRpcSettings.multicallContractAddress +
                "\",\"data\":\"" + probeData +
                "\"},\"latest\"]}";
            const std::string probeResultHex = GetRequiredRpcResultHex(client.PostJson(probeRequest));
            const std::vector<std::string> decodedProbe = DecodeMulticallAggregateResultsHex(probeResultHex, 2);
            if (decodedProbe.size() != 2) {
                error = "Invalid Multicall2 response during initial probe.";
                return false;
            }

            return true;
        } catch (const std::exception& ex) {
            error = ex.what();
            return false;
        }
    }

    void PrintFoundWallet(
        int workerId,
        const WalletEntry& wallet,
        const std::string& ethBalanceHex,
        const std::string& usdcBalanceHex,
        const std::string& usdtBalanceHex)
    {
        std::lock_guard<std::mutex> lock(gPrintMutex);
        const std::string privateKeyHex = "0x" + BytesToHexLower(wallet.privateKey.data(), wallet.privateKey.size());
        const std::string eth = FormatWeiHexAsEth(ethBalanceHex);
        const std::string usdc = FormatHexAmount(usdcBalanceHex, kUsdcDecimals);
        const std::string usdt = FormatHexAmount(usdtBalanceHex, kUsdtDecimals);

        std::cout
            << "[FOUND][w" << workerId << "]"
            << " addr=" << wallet.address
            << " | priv=" << privateKeyHex
            << " | wei=" << ethBalanceHex
            << " | eth=" << eth
            << " | usdc=" << usdc
            << " | usdt=" << usdt
            << "\n";
        std::cout.flush();

        std::ofstream out(kFoundWalletsFile, std::ios::app);
        if (out) {
            out
                << "Address: " << wallet.address << "\n"
                << "Private key: " << privateKeyHex << "\n"
                << "ETH: " << eth << "\n"
                << "USDC: " << usdc << "\n"
                << "USDT: " << usdt << "\n\n";
        }
    }

    struct PendingBatch
    {
        std::vector<WalletEntry> wallets;
        MicrosecondClock::time_point sentAt;
        std::size_t requestCount = 0;
        TokenRpcMode tokenRpcMode = TokenRpcMode::DirectCalls;
        std::vector<MulticallChunkRequest> multicallChunks;
    };

    void WorkerLoop(
        int workerId,
        const std::wstring& rpcUrl,
        const std::string& forcedAddress,
        bool enableCudaHashing,
        const RuntimeConfig& config,
        const TokenRpcSettings& tokenRpcSettings,
        RuntimeStats& stats)
    {
        try {
            WinHttpPersistentClient client(rpcUrl, config.requestTimeoutMs);
            std::deque<std::pair<std::future<std::string>, PendingBatch>> inflightRequests;

            auto processBatchResponse = [&](const std::string& response, const PendingBatch& batch) {
                auto t_parse_start = MicrosecondClock::now();
                const auto balances = ParseRpcResultsById(response, batch.requestCount);

                std::vector<std::string> usdcBalances;
                std::vector<std::string> usdtBalances;
                if (batch.tokenRpcMode == TokenRpcMode::Multicall2Aggregate) {
                    usdcBalances.assign(batch.wallets.size(), "0x0");
                    usdtBalances.assign(batch.wallets.size(), "0x0");

                    for (const MulticallChunkRequest& chunk : batch.multicallChunks) {
                        const std::string& aggregateHex = GetBatchResultHexOrZero(balances, chunk.requestId);
                        const std::vector<std::string> decodedChunk =
                            DecodeMulticallAggregateResultsHex(aggregateHex, chunk.walletCount * 2);

                        for (std::size_t i = 0; i < chunk.walletCount; ++i) {
                            const std::size_t walletIndex = chunk.walletStartIndex + i;
                            usdcBalances[walletIndex] = decodedChunk[i * 2];
                            usdtBalances[walletIndex] = decodedChunk[(i * 2) + 1];
                        }
                    }
                }

                auto t_parse_end = MicrosecondClock::now();
                stats.timeParseBatch.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t_parse_end - t_parse_start).count(), std::memory_order_relaxed);

                for (std::size_t i = 0; i < batch.wallets.size(); ++i) {
                    const WalletEntry& wallet = batch.wallets[i];
                    const std::string& ethBalanceHex = GetBatchResultHexOrZero(balances, wallet.ethRequestId);
                    const std::string& usdcBalanceHex =
                        (batch.tokenRpcMode == TokenRpcMode::Multicall2Aggregate)
                            ? usdcBalances[i]
                            : GetBatchResultHexOrZero(balances, wallet.usdcRequestId);
                    const std::string& usdtBalanceHex =
                        (batch.tokenRpcMode == TokenRpcMode::Multicall2Aggregate)
                            ? usdtBalances[i]
                            : GetBatchResultHexOrZero(balances, wallet.usdtRequestId);

                    const bool hasBalance =
                        !IsZeroHexAmount(ethBalanceHex) ||
                        !IsZeroHexAmount(usdcBalanceHex) ||
                        !IsZeroHexAmount(usdtBalanceHex);
                    if (!hasBalance) {
                        continue;
                    }

                    stats.foundWallets.fetch_add(1, std::memory_order_relaxed);
                    PrintFoundWallet(workerId, wallet, ethBalanceHex, usdcBalanceHex, usdtBalanceHex);
                }

                stats.totalWallets.fetch_add(batch.wallets.size(), std::memory_order_relaxed);
                stats.totalBatches.fetch_add(1, std::memory_order_relaxed);
            };

            while (!gStopRequested.load(std::memory_order_relaxed)) {
                try {
                    bool madeProgress = false;

                    // Process completed requests and collect timing
                    while (!inflightRequests.empty() && 
                           (inflightRequests.front().first.wait_for(std::chrono::seconds(0)) == std::future_status::ready || 
                            inflightRequests.size() >= static_cast<size_t>(config.inflightRequests))) {
                        
                        auto& front = inflightRequests.front();
                        try {
                            const std::string response = front.first.get();
                            const auto t_response_ready = MicrosecondClock::now();
                            stats.timeHttpRoundTrip.fetch_add(
                                std::chrono::duration_cast<std::chrono::microseconds>(t_response_ready - front.second.sentAt).count(),
                                std::memory_order_relaxed);
                            processBatchResponse(response, front.second);
                        } catch (const std::exception& ex) {
                            stats.totalErrors.fetch_add(1, std::memory_order_relaxed);
                            std::lock_guard<std::mutex> lock(gPrintMutex);
                            std::cerr << "[worker " << workerId << "] response error: " << ex.what() << "\n";
                        }
                        inflightRequests.pop_front();
                        madeProgress = true;
                    }

                    // Generate new batch and send if slots available
                    if (inflightRequests.size() < static_cast<size_t>(config.inflightRequests)) {
                        auto t0 = MicrosecondClock::now();
                        WalletBuildMetrics buildMetrics;
                        auto wallets = BuildRandomWallets(config.walletsPerBatch, forcedAddress, enableCudaHashing, buildMetrics);
                        auto t1 = MicrosecondClock::now();
                        BatchRequestBuildResult batchRequest = BuildBatchRequest(wallets, tokenRpcSettings);
                        auto t2 = MicrosecondClock::now();

                        stats.timeGenWallets.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), std::memory_order_relaxed);
                        stats.timeRandomPrivateKeys.fetch_add(buildMetrics.randomFillUs, std::memory_order_relaxed);
                        stats.timePublicKeyDerivation.fetch_add(buildMetrics.publicKeyDerivationUs, std::memory_order_relaxed);
                        stats.timeAddressHash.fetch_add(buildMetrics.addressHashUs, std::memory_order_relaxed);
                        stats.timeBuildRequest.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(), std::memory_order_relaxed);
                        if (buildMetrics.hashedAddresses) {
                            if (buildMetrics.usedGpu) {
                                stats.totalCudaHashBatches.fetch_add(1, std::memory_order_relaxed);
                            } else {
                                stats.totalCpuHashBatches.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        if (buildMetrics.gpuFallback) {
                            stats.totalCudaFallbackBatches.fetch_add(1, std::memory_order_relaxed);
                            if (!buildMetrics.gpuError.empty()) {
                                std::call_once(gCudaFallbackPrintOnce, [&buildMetrics]() {
                                    std::lock_guard<std::mutex> lock(gPrintMutex);
                                    std::cerr << "[CUDA] falling back to CPU for secp256k1+keccak pipeline: " << buildMetrics.gpuError << "\n";
                                    });
                            }
                        }

                        const auto sentAt = MicrosecondClock::now();
                        auto future = std::async(std::launch::async, [&client, req = std::move(batchRequest.json)]() {
                            return client.PostJson(req);
                        });

                        PendingBatch pending{
                            std::move(wallets),
                            sentAt,
                            batchRequest.requestCount,
                            tokenRpcSettings.mode,
                            std::move(batchRequest.multicallChunks)
                        };
                        inflightRequests.emplace_back(std::move(future), std::move(pending));
                        madeProgress = true;
                    }

                    if (!madeProgress) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                }
                catch (const std::exception& ex) {
                    stats.totalErrors.fetch_add(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(gPrintMutex);
                    std::cerr << "[worker " << workerId << "] loop error: " << ex.what() << "\n";
                }
            }

            // Drain remaining requests
            while (!inflightRequests.empty()) {
                try {
                    const std::string response = inflightRequests.front().first.get();
                    const auto t_response_ready = MicrosecondClock::now();
                    stats.timeHttpRoundTrip.fetch_add(
                        std::chrono::duration_cast<std::chrono::microseconds>(t_response_ready - inflightRequests.front().second.sentAt).count(),
                        std::memory_order_relaxed);
                    processBatchResponse(response, inflightRequests.front().second);
                } catch (const std::exception&) {
                    stats.totalErrors.fetch_add(1, std::memory_order_relaxed);
                }
                inflightRequests.pop_front();
            }
        }
        catch (const std::exception& ex) {
            stats.totalErrors.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(gPrintMutex);
            std::cerr << "[worker " << workerId << "] client initialization error: " << ex.what() << "\n";
        }
    }

    BOOL WINAPI ConsoleCtrlHandler(DWORD signalType)
    {
        switch (signalType) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            gStopRequested.store(true, std::memory_order_relaxed);
            return TRUE;
        default:
            return FALSE;
        }
    }
} // namespace

int wmain()
{
    try {
        const RuntimeConfig config = BuildRuntimeConfig();
        const std::wstring rpcUrl = kDefaultRpcUrl;
        const int workerCount = config.workerCount;
        if (workerCount < 1) {
            throw std::runtime_error("CONFIG_WORKER_THREADS must be >= 1.");
        }
        if (config.walletsPerBatch < 1) {
            throw std::runtime_error("CONFIG_WALLETS_PER_BATCH must be >= 1.");
        }
        if (config.inflightRequests < 1) {
            throw std::runtime_error("CONFIG_INFLIGHT_REQUESTS must be >= 1.");
        }

        std::string forcedAddress;
        constexpr const char* kConfiguredForceAddress = CONFIG_FORCE_ADDRESS;
        if (kConfiguredForceAddress[0] != '\0') {
            forcedAddress = NormalizeAddressOrThrow(kConfiguredForceAddress);
        }

        bool enableCudaHashing = false;
        std::string addressHashBackend = "CPU (build without CUDA)";
#ifdef ETH_USE_CUDA
        const eth_cuda::CudaDeviceInfo cudaInfo = eth_cuda::QueryCudaDeviceInfo();
        if (cudaInfo.available) {
            std::string cudaValidationError;
            if (ValidateCudaAddressPipeline(cudaValidationError)) {
                enableCudaHashing = true;
                addressHashBackend =
                    "CUDA secp256k1 + keccak: " + cudaInfo.name +
                    " (sm_" + std::to_string(cudaInfo.computeMajor) + std::to_string(cudaInfo.computeMinor) + ")";
            } else {
                addressHashBackend = "CPU fallback: " + cudaValidationError;
            }
        } else if (!cudaInfo.error.empty()) {
            addressHashBackend = "CPU fallback: " + cudaInfo.error;
        } else {
            addressHashBackend = "CPU fallback: CUDA runtime not available";
        }
#endif

        TokenRpcSettings tokenRpcSettings;
        tokenRpcSettings.multicallContractAddress = NormalizeAddressOrThrow(kMulticall2ContractAddress);
        tokenRpcSettings.walletsPerMulticall = config.multicallWalletsPerCall;
        std::string multicallProbeError;
        if (ProbeMulticall2Support(rpcUrl, config.requestTimeoutMs, tokenRpcSettings, multicallProbeError)) {
            tokenRpcSettings.mode = TokenRpcMode::Multicall2Aggregate;
            tokenRpcSettings.backendDescription =
                "Multicall2 aggregate via eth_call: " + tokenRpcSettings.multicallContractAddress +
                " | wallets/chunk=" + std::to_string(tokenRpcSettings.walletsPerMulticall);
        } else {
            tokenRpcSettings.backendDescription =
                "Direct eth_call for each token (fallback): " + multicallProbeError;
        }

        if (!SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE)) {
            throw std::runtime_error("Unable to register Ctrl+C handler.");
        }

        RuntimeStats stats;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(workerCount));

        for (int i = 0; i < workerCount; ++i) {
            workers.emplace_back(
                WorkerLoop,
                i + 1,
                std::cref(rpcUrl),
                std::cref(forcedAddress),
                enableCudaHashing,
                std::cref(config),
                std::cref(tokenRpcSettings),
                std::ref(stats));
        }

        {
            std::lock_guard<std::mutex> lock(gPrintMutex);
            std::cout << "RPC URL: " << Narrow(rpcUrl) << "\n";
            std::cout << "Wallets per batch: " << config.walletsPerBatch << "\n";
            std::cout << "Worker threads: " << workerCount << "\n";
            std::cout << "Inflight requests/worker: " << config.inflightRequests << "\n";
            std::cout << "Address pipeline: " << addressHashBackend << "\n";
            std::cout << "Token query pipeline: " << tokenRpcSettings.backendDescription << "\n";
#ifdef ETH_USE_CUDA
            if (enableCudaHashing) {
                std::cout
                    << "CUDA batching: max_keys=" << kGpuHashMaxBatchKeys
                    << " | flush_us=" << kGpuHashBatchFlushDelay.count()
                    << "\n";
            }
#endif
            if (!forcedAddress.empty()) {
                std::cout << "FORCE_ADDRESS: " << forcedAddress << "\n";
            }
            std::cout << "Continuous loop started. Press Ctrl+C to stop.\n";
            std::cout.flush();
        }

        std::uint64_t lastWallets = 0;
        std::uint64_t lastBatches = 0;
        auto lastTime = std::chrono::steady_clock::now();

        while (!gStopRequested.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(config.statsIntervalSeconds));

            const auto now = std::chrono::steady_clock::now();
            const double elapsedSeconds = std::chrono::duration<double>(now - lastTime).count();
            if (elapsedSeconds <= 0.0) {
                continue;
            }

            const std::uint64_t currentWallets = stats.totalWallets.load(std::memory_order_relaxed);
            const std::uint64_t currentBatches = stats.totalBatches.load(std::memory_order_relaxed);
            const std::uint64_t currentErrors = stats.totalErrors.load(std::memory_order_relaxed);
            const std::uint64_t currentFound = stats.foundWallets.load(std::memory_order_relaxed);

            const double walletsPerSecond = static_cast<double>(currentWallets - lastWallets) / elapsedSeconds;
            const double batchesPerSecond = static_cast<double>(currentBatches - lastBatches) / elapsedSeconds;

            {
                std::lock_guard<std::mutex> lock(gPrintMutex);
                const std::uint64_t batchCount = currentBatches > 0 ? currentBatches : 1;
                const std::uint64_t hashBatchCount =
                    stats.totalCudaHashBatches.load(std::memory_order_relaxed) +
                    stats.totalCpuHashBatches.load(std::memory_order_relaxed);
                const double averageHashMs =
                    hashBatchCount > 0 ?
                    (stats.timeAddressHash.load(std::memory_order_relaxed) / 1000.0 / static_cast<double>(hashBatchCount)) :
                    0.0;
#ifdef ETH_USE_CUDA
                const std::uint64_t gpuDispatchCount = gGpuDispatchBatches.load(std::memory_order_relaxed);
                const std::uint64_t gpuDispatchKeys = gGpuDispatchKeys.load(std::memory_order_relaxed);
                const double averageGpuKeysPerDispatch =
                    gpuDispatchCount > 0 ?
                    (static_cast<double>(gpuDispatchKeys) / static_cast<double>(gpuDispatchCount)) :
                    0.0;
                const double averageWorkerBatchesPerDispatch =
                    gpuDispatchCount > 0 ?
                    (static_cast<double>(stats.totalCudaHashBatches.load(std::memory_order_relaxed)) / static_cast<double>(gpuDispatchCount)) :
                    0.0;
#endif
                
                std::cout
                    << "[STATS] total_wallets=" << currentWallets
                    << " | total_batches=" << currentBatches
                    << " | wallets/s=" << std::fixed << std::setprecision(2) << walletsPerSecond
                    << " | batches/s=" << std::fixed << std::setprecision(2) << batchesPerSecond
                    << " | found=" << currentFound
                    << " | errors=" << currentErrors
                    << " || avg_ms per batch:"
                    << " gen=" << std::fixed << std::setprecision(2) << (stats.timeGenWallets.load() / 1000.0 / batchCount)
                    << " priv=" << (stats.timeRandomPrivateKeys.load() / 1000.0 / batchCount)
                    << " pub_cpu=" << (stats.timePublicKeyDerivation.load() / 1000.0 / batchCount)
                    << " addr_pipeline=" << averageHashMs
                    << " build=" << (stats.timeBuildRequest.load() / 1000.0 / batchCount)
                    << " http=" << (stats.timeHttpRoundTrip.load() / 1000.0 / batchCount)
                    << " parse=" << (stats.timeParseBatch.load() / 1000.0 / batchCount)
                    << " | addr_gpu_batches=" << stats.totalCudaHashBatches.load(std::memory_order_relaxed)
                    << " | addr_cpu_batches=" << stats.totalCpuHashBatches.load(std::memory_order_relaxed)
                    << " | cuda_fallbacks=" << stats.totalCudaFallbackBatches.load(std::memory_order_relaxed)
#ifdef ETH_USE_CUDA
                    << " | gpu_dispatches=" << gpuDispatchCount
                    << " | avg_gpu_keys/dispatch=" << averageGpuKeysPerDispatch
                    << " | avg_req_batches/dispatch=" << averageWorkerBatchesPerDispatch
#endif
                    << "\n";
                std::cout.flush();
            }

            lastWallets = currentWallets;
            lastBatches = currentBatches;
            lastTime = now;
        }

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

#ifdef ETH_USE_CUDA
        if (enableCudaHashing) {
            GetGpuAddressHashService().Shutdown();
        }
#endif

        {
            std::lock_guard<std::mutex> lock(gPrintMutex);
            std::cout << "Shutdown completed.\n";
            std::cout.flush();
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
