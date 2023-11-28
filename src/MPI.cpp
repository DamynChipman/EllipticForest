#include "MPI.hpp"

namespace EllipticForest {

namespace MPI {

// void communicatorUnion(std::vector<Communicator> comms, Communicator* comm_union) {
//     // Get the groups from the communicators
//     std::vector<Group> groups;
//     for (auto& comm : comms) {
//         Group group
//         MPI_Comm_group(comm, &group);
//         groups.push_back(group)
//     }

//     // Iterate over groups and combine
//     Group unioned_group;
//     for (auto& group : groups) {
//         MPI_Group_union(unioned_group, group, &unioned_group);
//     }

//     // Create new communicator
//     MPI_Comm_create_group()
// }

void communicatorSubsetRange(Communicator super_comm, int r_first, int r_last, int tag, Communicator* sub_comm) {

    // // Get group of super communicator
    // Group super_group;
    // MPI_Comm_group(super_comm, &super_group);

    // // Create new sub group from super group
    // Group sub_group;
    // int ranges[1][3] = {r_first, r_last, 1};
    // MPI_Group_range_incl(super_group, 1, ranges, &sub_group);

    // // Create sub communicator from sub group
    // MPI_Comm_create_group(super_comm, sub_group, tag, sub_comm);

    // // Set name of comm
    // int super_comm_name_size = 0;
    // char super_comm_name_buffer[255];
    // MPI_Comm_get_name(super_comm, super_comm_name_buffer, &super_comm_name_size);
    // std::string super_comm_name(super_comm_name_buffer, super_comm_name_size);
    // std::string sub_comm_name = super_comm_name + "_sub_range={" + std::to_string(r_first) + ":" + std::to_string(r_last) + "}";
    // MPI_Comm_set_name(*sub_comm, sub_comm_name.c_str());

    // // Free groups
    // MPI_Group_free(&super_group);
    // MPI_Group_free(&sub_group);

    int rank; MPI_Comm_rank(super_comm, &rank);
    MPI_Comm_split(super_comm, (r_first <= rank && rank <= r_last ? 0 : MPI_UNDEFINED), rank, sub_comm);

}

void communicatorSetName(Communicator comm, const std::string& name) {
    MPI_Comm_set_name(comm, name.c_str());
}

std::string communicatorGetName(Communicator comm) {
    int comm_name_size = 0;
    char comm_name_buffer[255];
    MPI_Comm_get_name(comm, comm_name_buffer, &comm_name_size);
    return std::string(comm_name_buffer, comm_name_size);
}

template<>
int send<std::string>(std::string& str, int dest, int tag, MPI_Comm comm) {
    int size = str.length();
    send(size, dest, tag, comm);
    return MPI_Send(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), dest, tag, comm);
}

template<>
int receive<std::string>(std::string& str, int src, int tag, MPI_Comm comm, MPI_Status* status) {
    int size;
    receive(size, src, tag, comm, status);
    str.resize(size);
    return MPI_Recv(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), src, tag, comm, status);
}

template<>
int broadcast<std::string>(std::string& str, int root, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    int size = str.length();
    broadcast(size, root, comm);
    if (rank != root) str = std::string(TypeTraits<std::string>::getAddress(str), size);
    return MPI_Bcast(TypeTraits<std::string>::getAddress(str), TypeTraits<std::string>::getSize(str), TypeTraits<std::string>::getType(str), root, comm);
}

} // NAMESPACE : MPI

} // NAMESPACE : EllipticForest