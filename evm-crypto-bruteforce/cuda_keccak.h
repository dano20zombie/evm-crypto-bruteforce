#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace eth_cuda
{
    struct CudaDeviceInfo
    {
        bool available = false;
        int deviceCount = 0;
        int selectedDevice = 0;
        int computeMajor = 0;
        int computeMinor = 0;
        std::string name;
        std::string error;
    };

    CudaDeviceInfo QueryCudaDeviceInfo();

    bool ComputeEthereumAddressesFromPrivateKeysCuda(
        const std::uint8_t* privateKeys,
        std::size_t keyCount,
        std::uint8_t* outAddressBytes,
        std::string& error);

    bool ComputeEthereumAddressHashesCuda(
        const std::uint8_t* publicKeysNoPrefix,
        std::size_t keyCount,
        std::uint8_t* outAddressBytes,
        std::string& error);
}
