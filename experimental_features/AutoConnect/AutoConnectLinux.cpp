//
// Created by magnus on 7/14/22.
//

#include <iostream>
#include <cstring>
#include <linux/sockios.h>
#include <net/if.h>
#include <execution>
#include <linux/ethtool.h>
#include <netinet/ip.h>
#include <sys/ioctl.h>
#include <MultiSense/details/channel.hh>
#include <arpa/inet.h>
#include "AutoConnectLinux.h"


std::vector<AutoConnect::AdapterSupportResult> AutoConnectLinux::findEthernetAdapters() {
    std::vector<AdapterSupportResult> adapterSupportResult;
    auto ifn = if_nameindex();
    auto fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    for (auto i = ifn; i->if_name; ++i) {
        struct {
            struct ethtool_link_settings req;
            __u32 link_mode_data[3 * 127];
        } ecmd{};

        adapterSupportResult.emplace_back(i->if_name, false);

        // Skip the loopback
        //if (i->if_index == 1) {
        //continue;
        //}

        auto ifr = ifreq{};
        std::strncpy(ifr.ifr_name, i->if_name, IF_NAMESIZE);

        ecmd.req.cmd = ETHTOOL_GLINKSETTINGS;
        ifr.ifr_data = reinterpret_cast<char *>(&ecmd);

        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            std::cerr << "Skipping Adapter: " << i->if_name << "| Reason: ioctl fail | " << strerror(errno)
                      << std::endl;
            continue;

        }

        if (ecmd.req.link_mode_masks_nwords >= 0 || ecmd.req.cmd != ETHTOOL_GLINKSETTINGS) {
            std::cerr << "Skipping Adapter: " << i->if_name << "| Reason: link_mode != ETHTOOL_GLINKSETTINGS | "
                      << std::endl;
            continue;
        }

        ecmd.req.link_mode_masks_nwords = -ecmd.req.link_mode_masks_nwords;

        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            std::cerr << "Skipping Adapter: " << i->if_name << "| Reason: ioctl fail | " << strerror(errno)
                      << std::endl;
            continue;
        }

        std::cout << "\n\n\tFound Adapter: " << i->if_name
                  << "\n\tSpeed: " << ecmd.req.speed
                  << "\n\tDuplex: " << static_cast<int>(ecmd.req.duplex)
                  << "\n\tPort: " << static_cast<int>(ecmd.req.port)
                  << std::endl;

        adapterSupportResult.back().supports = true;
    }

    if (!adapterSupportResult.empty())
        onFoundAdapters(adapterSupportResult);
    return adapterSupportResult;
}

void AutoConnectLinux::run(void *app, std::vector<AdapterSupportResult> adapters) {
    AutoConnectLinux *instance = (AutoConnectLinux *) app;

    // Get list of network adapters that are  supports our application
    std::string hostAddress;
    int i = 0;

    // Loop keeps retrying to connect on supported network adapters.
    while (instance->loopAdapters) {
        auto adapter = adapters[i];
        i++;
        if (i == adapters.size())
            i = 0;

        if (!adapter.supports) {
            continue;
        }
        printf("\nTesting Adapter: %s\n", adapter.name.c_str());
        int sd = -1;
        // Submit request for a socket descriptor to look up interface.
        if ((sd = socket(PF_INET, SOCK_RAW, IPPROTO_RAW)) < 0) {
            perror("socket() failed to get socket descriptor for using ioctl() ");
            exit(EXIT_FAILURE);
        }

        /* set the network card in promiscuos mode*/
        // An ioctl() request has encoded in it whether the argument is an in parameter or out parameter
        // SIOCGIFFLAGS	0x8913		/* get flags			*/
        // SIOCSIFFLAGS	0x8914		/* set flags			*/
        struct ifreq ethreq;
        strncpy(ethreq.ifr_name, adapter.name.c_str(), IF_NAMESIZE);
        if (ioctl(sd, SIOCGIFFLAGS, &ethreq) == -1) {
            perror("ioctl");
            close(sd);
            exit(1);
        }
        ethreq.ifr_flags |= IFF_PROMISC;
        if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
            perror("ioctl");
            close(sd);
            exit(1);
        }

        int saddr_size, data_size;
        struct sockaddr saddr{};
        auto *buffer = (unsigned char *) malloc(IP_MAXPACKET + 1);

        int sock_raw = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
        if (sock_raw < 0) {
            //Print the error with proper message
            perror("Socket Error");
            return;
        }
        int ret = setsockopt(sock_raw, SOL_SOCKET, SO_BINDTODEVICE, adapter.name.c_str(), adapter.name.length() + 1);
        if (ret != 0) {
            std::cerr << "Failed to bind to network adapter" << std::endl;
            continue;
        }


        printf("Listening for a IGMP packet\n");
        while (instance->listenOnAdapter) {
            saddr_size = sizeof saddr;
            //Receive a packet
            data_size = (int) recvfrom(sock_raw, buffer, IP_MAXPACKET, 0, &saddr,
                                       (socklen_t *) &saddr_size);

            if (data_size < 0) {
                printf("Recvfrom error , failed to get packets\n");
                return;
            }
            //Now process the packet
            //ProcessPacket(buffer, data_size, &ipAddress);
            auto *iph = (struct iphdr *) (buffer + sizeof(struct ethhdr));
            struct in_addr ip_addr{};
            std::string address;
            if (iph->protocol == IPPROTO_IGMP) //Check the Protocol and do accordingly...
            {
                ip_addr.s_addr = iph->saddr;
                address = inet_ntoa(ip_addr);
                printf("Packet found. Source address: %s\n", address.c_str());

                if (instance->onFoundIp(address, adapter)) {
                    instance->loopAdapters = false;
                    break;
                }

            }
        }

    }

    printf("Exited thread\n");
}

void AutoConnectLinux::onFoundAdapters(std::vector<AdapterSupportResult> adapters) {

}

bool AutoConnectLinux::onFoundIp(std::string address, AdapterSupportResult adapter) {
// Set the host ip address to the same subnet but with *.1 at the end.
    std::string hostAddress = address;
    std::string last_element(hostAddress.substr(hostAddress.rfind(".")));
    auto ptr = hostAddress.rfind(".");
    hostAddress.replace(ptr, last_element.length(), ".1");
    printf("Setting host address to: %s\n", hostAddress.c_str());



    /*** CALL IOCTL Operations to set the address of the adapter/socket  ***/
    // Create the socket.
    int camera_fd = -1;
    camera_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (camera_fd < 0)
        fprintf(stderr, "failed to create the UDP socket: %s",
                strerror(errno));

    // Bind Camera FD to the ethernet device
    const char *interface = adapter.name.c_str();
    int ret = setsockopt(camera_fd, SOL_SOCKET, SO_BINDTODEVICE, interface,
                         IFNAMSIZ); // 15 is max length for an adapter name.
    if (ret != 0) {
        fprintf(stderr, "Error binding to: %s, %s", interface, strerror(errno));
    }

    struct ifreq ifr{};
    /// note: no pointer here
    struct sockaddr_in inet_addr{}, subnet_mask{};
    /* get interface name */
    /* Prepare the struct ifreq */
    bzero(ifr.ifr_name, IFNAMSIZ);
    strncpy(ifr.ifr_name, interface, IFNAMSIZ);

    /// note: prepare the two struct sockaddr_in
    inet_addr.sin_family = AF_INET;
    int inet_addr_config_result = inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

    subnet_mask.sin_family = AF_INET;
    int subnet_mask_config_result = inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));


    // Call ioctl to configure network devices
    /// put addr in ifr structure
    memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
    int ioctl_result = ioctl(camera_fd, SIOCSIFADDR, &ifr);  // Set IP address
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFADDR: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /// put mask in ifr structure
    memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
    ioctl_result = ioctl(camera_fd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFNETMASK: ");
        perror("");
        exit(EXIT_FAILURE);
    }
    /*** END **/

    // Attempt to connect to camera and post some info
    auto *cameraInterface = crl::multisense::Channel::Create(address);

    if (cameraInterface != nullptr) {
        onFoundCamera();
        return true;
    } else {
        printf("Did not find a camera on %s\n Retrying...\n", address.c_str());
        close(camera_fd);
    }
    return false;
}

void AutoConnectLinux::onFoundCamera() {
    success = true;
}

void AutoConnectLinux::stop() {
    loopAdapters = false;
    listenOnAdapter = false;
    printf("Joining thread\n");
    t->join();
}

void AutoConnectLinux::start(std::vector<AdapterSupportResult> adapters) {
    t = new std::thread(&AutoConnectLinux::run, this, adapters);
    printf("Started thread\n");

}
