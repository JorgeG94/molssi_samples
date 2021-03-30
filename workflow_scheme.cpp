/**
This routine, alongside the the .hpp file contains a simple template 
abstraction for the general workflow scheme used in the code. This
workflow scheme is based on a coordinator worker algorithm. The mpi
rank == 0 contains the master, which dynamically distributes work 
to the workers mpi_rank > 0.  
This function is expandable to different routines, with different 
types of functions and/or types of functions, such as two center 
(one electron integrals), three center (RI integrals) and four center
integrals (ERI + Digestion)
*/

#include "workflow_scheme.hpp"


// template specialization for the 4 center work in scf.cpp
template < > 
void Workflow_scheme<genfock::Quartet_Batch_Generator_Sym,
                     genfock::GPU_BraBatch_Bin&,
                     int,
                     std::vector<genfock::ShellQuartetType>&,
                     genfock::CUDA_GPU::DeviceKit>::workflow
                     (genfock::Quartet_Batch_Generator_Sym index_gen,
                     genfock::GPU_BraBatch_Bin& shell_container1,
                     int i,
                     std::vector<genfock::ShellQuartetType>& n_et_types,
                     genfock::CUDA_GPU::DeviceKit &gpu_kit,
                     MPI_Comm local_mpi_comm){
    int mpi_world_size, mpi_rank;
    MPI_Comm_size(local_mpi_comm, &mpi_world_size);
    MPI_Comm_rank(local_mpi_comm, &mpi_rank);
    //------MASTER PROCESS------//
    // Following is code executed by only the master MPI process
    if (mpi_rank == 0) {
      try {
        //-------CREATE QUARTET BATCH GENERATOR-----------//

        bool first_cycle(true);
        bool break_at_first_cycle(false);
        int n_active_processes(0);
        unsigned n_ets_sent(0);
        unsigned n_ets_recv(0);
        while (true) {
          //--------SEND INITIAL QUARTET BATCHES---------//
          if (first_cycle == true) {
            int iproc(1);
            while (iproc < mpi_world_size) {
              int *initial_indices;
              initial_indices = index_gen.get_new_indices(
                  shell_container1, n_et_types, m_ket_max_batch_multiple,
                  m_dynamic_threshold);

              int bra_type_id_index = index_gen.get_bra_type_id_index();
              int n_bra_batches_index = index_gen.get_n_bra_batches_index();
              int n_ket_batches_index = index_gen.get_n_ket_batches_index();

              int i_bra_type_id = initial_indices[bra_type_id_index];
              int i_n_bra_batches = initial_indices[n_bra_batches_index];
              int i_n_ket_batches = initial_indices[n_ket_batches_index];
              if (i_bra_type_id == -1) {
                break;
              }

              if ((i_n_ket_batches == 0) || (i_n_bra_batches == 0)) {
                continue;
              }

              // ACCUMULATES SHELL QUATETS
              MPI_Send(initial_indices, 9, MPI_INT, iproc, 1, local_mpi_comm);
              ++n_ets_sent;
              ++iproc;
            }
            if ((iproc - 1) != mpi_world_size - 1) {
              n_active_processes = iproc;
              break_at_first_cycle = true;
            }
            first_cycle = false;
          }
          if (!break_at_first_cycle) {
            //-------SEND REMAINING QUARTET BATCHES DYNAMICALLY--------//
            int *nindices;
            nindices = index_gen.get_new_indices(shell_container1, n_et_types,
                                            m_ket_max_batch_multiple,
                                            m_dynamic_threshold);

            int bra_type_id_index = index_gen.get_bra_type_id_index();
            int n_bra_batches_index = index_gen.get_n_bra_batches_index();
            int n_ket_batches_index = index_gen.get_n_ket_batches_index();

            int bra_type_id = nindices[bra_type_id_index];
            int n_bra_batches = nindices[n_bra_batches_index];
            int n_ket_batches = nindices[n_ket_batches_index];

            if (bra_type_id == -1) {
              //------WORK DISTRIBUTION COMPLETED - WRAP UP------//
              // Wait for all processes to finish calculation
              // std::cout << "BEFORE FINAL GATHER!"<< std::endl;
              while (n_ets_sent > n_ets_recv) {

                MPI_Status status;
                int islave;
                int isource;

                MPI_Probe(MPI_ANY_SOURCE, 1, local_mpi_comm, &status);
                isource = status.MPI_SOURCE;
                MPI_Recv(&islave, 1, MPI_INT, isource, 1, local_mpi_comm, &status);
                ++n_ets_recv;
              }
              for (int iproc = 1; iproc < mpi_world_size; ++iproc) {
                // Send end signal to iproc
                int end_message(0);
                MPI_Send(&end_message, 1, MPI_INT, iproc, 0, local_mpi_comm);
              }

              break;
            } else {

              if ((n_ket_batches == 0) || (n_bra_batches == 0)) {
                continue;
              }
              // Wait to receive signal from any slave process
              MPI_Status status;
              int isource;
              int islave;

              MPI_Probe(MPI_ANY_SOURCE, 1, local_mpi_comm, &status);
              isource = status.MPI_SOURCE;
              MPI_Recv(&islave, 1, MPI_INT, isource, 1, local_mpi_comm, &status);
              ++n_ets_recv;

              // islave is free, give it some work to do

              MPI_Send(nindices, 9, MPI_INT, islave, 1, local_mpi_comm);
              ++n_ets_sent;
            }
          } else if (break_at_first_cycle) {
            // MORE PROCESSES THAN QUARTET BATCHES
            // Wait for all processes to finish calculation
            std::cout
                << "WARNING: there are more processes that quartet batches!"
                << std::endl;
            for (int iproc = 1; iproc < n_active_processes; ++iproc) {

              MPI_Status status;
              int islave;
              MPI_Recv(&islave, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, local_mpi_comm,
                       &status);
              ++n_ets_recv;
              // Send end signal to islave
              int end_message(0);
              MPI_Send(&end_message, 1, MPI_INT, islave, 0, local_mpi_comm);
              ++n_ets_sent;
              break;
            }
          }
        }
      } catch (...) {

        throw;
      }
    } else if (mpi_rank > 0) {
      try {
        //------SLAVE PROCESSES------//
        int nindices[9];
        int tag;
        int end_message;
        MPI_Status status;

        MPI_Probe(0, 1, local_mpi_comm, &status);
        tag = status.MPI_TAG;

        while (tag == 1) {
          MPI_Recv(nindices, 9, MPI_INT, 0, 1, local_mpi_comm, &status);

          //-----GPU WORK HERE--------//
        
          // Compute and digest integrals into the fock matrix
          gpu_kit.form_and_digest_eris(shell_container1, nindices);

          MPI_Send(&mpi_rank, 1, MPI_INT, 0, 1, local_mpi_comm);

          MPI_Probe(0, MPI_ANY_TAG, local_mpi_comm, &status);
          tag = status.MPI_TAG;
          if (tag == 0) {
            MPI_Recv(&end_message, 1, MPI_INT, 0, 0, local_mpi_comm, &status);
          }
        } // while (tag == 1)
      } catch (...) {
        gpu_kit.free_mem_pool();
        //cublasDestroy(blas_handle);

        throw;
      }
    } // else if (mpi_rank > 0) slave process
    
}
