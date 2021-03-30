/**
 * @file worflow_scheme.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-02-20
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef WORKFLOW_SCHEME_HPP
#define WORKFLOW_SCHEME_HPP

#include "mpi.h"

#include <vector>

#include "../basis/basis.hpp"
#include "../basis/shells.hpp"
#include "../batchbin/bin.hpp"
#include "../core/queue.hpp"
#include "../methods/hf/cuda_hf/device_kit.h"
/**
 * @brief Workflow_scheme class. Will create the queueing system used by the coordinator worker method
 * @param T1 doublet, triplet or quartet batch gen (for the indices)
 * @param T2 BraBatchBin or ShellBatchContainer
 * @param T3 BraBatchBin or ShellBatchContainer
 * @param T4 doublet, triplet or quartet types
 * @param T5 DeviceKit type
 */
template <class T1, class T2, class T3, class T4, class T5> 
class Workflow_scheme{

  const unsigned int m_ket_max_batch_multiple;
  int m_dynamic_threshold;
  int m_n_indices; // indicate how many indices are stored in the indices array


  public:

  /**
   * @brief Construct a new Workflow_scheme object
   * Default constructor 
   */
  Workflow_scheme(){}
  /**
  * @brief current parametrized constructor 
  *
  */
  Workflow_scheme(const unsigned int ket_max_batch_multiple, int dynamic_threshold, int n_indices)
                 : m_ket_max_batch_multiple(ket_max_batch_multiple),
                   m_dynamic_threshold(dynamic_threshold), m_n_indices(n_indices){};
  // for scf routine
  Workflow_scheme(const unsigned int ket_max_batch_multiple, int dynamic_threshold)
                 : m_ket_max_batch_multiple(ket_max_batch_multiple),
                   m_dynamic_threshold(dynamic_threshold){};

  ~Workflow_scheme(){};

  /**
   * @brief workflow general function
   * 
   */
  void workflow(T1 index_gen, T2& shell_container1, T3 shell_container2, 
                T4& n_et_types,
                T5 &gpu_kit,
                MPI_Comm local_mpi_comm
                );
  // template function
  /**
   * @brief workflow template for general routines
   * 
   */
  void workflow(T1 index_gen, T2& shell_container1, T3& shell_container2, 
                T4& n_et_types,
                T5 &gpu_kit,
                void (T5::*gpu_work)(T2&, T3&, int *),
                MPI_Comm local_mpi_comm
                ){

    // threshold is currently a place holder here 
    unsigned threshold = 1; 

    int mpi_world_size, mpi_rank;
    MPI_Comm_size(local_mpi_comm, &mpi_world_size);
    MPI_Comm_rank(local_mpi_comm, &mpi_rank);

    //------MASTER PROCESS------//
    // Following is code executed by only the master MPI process
    if (mpi_rank == 0) {
      try {
        //-------CREATE N-TET BATCH GENERATOR-----------//

        bool first_cycle(true);
        bool break_at_first_cycle(false);
        int n_active_processes(0);
        unsigned n_ets_sent(0);
        unsigned n_ets_recv(0);
        while (true) {
          //--------SEND INITIAL N-TET BATCHES---------//
          if (first_cycle == true) {
            int iproc(1);
            while (iproc < mpi_world_size) {
              int *initial_indices;
              initial_indices = index_gen.get_new_indices(
                  shell_container1, shell_container2, n_et_types,
                  threshold);

              // these here allow us to retrieve the indices from the respective container
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

              // ACCUMULATES SHELL N-TETS
              MPI_Send(initial_indices, m_n_indices, MPI_INT, iproc, 1, local_mpi_comm);
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
            //-------SEND REMAINING N-TET BATCHES DYNAMICALLY--------//
            int *nindices;
            nindices = index_gen.get_new_indices(shell_container1, shell_container2,
                                                 n_et_types,
                                                 threshold);

            int bra_type_id_index = index_gen.get_bra_type_id_index();
            int n_bra_batches_index = index_gen.get_n_bra_batches_index();
            int n_ket_batches_index = index_gen.get_n_ket_batches_index();

            int bra_type_id = nindices[bra_type_id_index];
            int n_bra_batches = nindices[n_bra_batches_index];
            int n_ket_batches = nindices[n_ket_batches_index];

            if (bra_type_id == -1) {
              //------WORK DISTRIBUTION COMPLETED - WRAP UP------//
              // Wait for all processes to finish calculation
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

              MPI_Send(nindices, m_n_indices, MPI_INT, islave, 1, local_mpi_comm);
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
          MPI_Recv(nindices, m_n_indices, MPI_INT, 0, 1, local_mpi_comm, &status);

        //-----GPU WORK HERE--------//
        
        // Compute and digest integrals into the fock matrix
        // call member function via the pointer 
        (gpu_kit.*gpu_work)(shell_container1, shell_container2, nindices);

          MPI_Send(&mpi_rank, 1, MPI_INT, 0, 1, local_mpi_comm);

          MPI_Probe(0, MPI_ANY_TAG, local_mpi_comm, &status);
          tag = status.MPI_TAG;
          if (tag == 0) {
            MPI_Recv(&end_message, 1, MPI_INT, 0, 0, local_mpi_comm, &status);
          }
        } // while (tag == 1)
      } catch (...) {
        // need to bring this bac when it is ready 

        gpu_kit.free_mem_pool();
        //cublasDestroy(blas_handle);

        throw;
      }
    } // else if (mpi_rank > 0) slave process
                
  } // workflow 

}; // Worfklow class



#endif
