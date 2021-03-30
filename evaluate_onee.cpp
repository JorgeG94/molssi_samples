/**
This is a general routine that calls the kernel evaluator for the one electron 
integrals. I wrote this routine to call the proper schedulers and taking the 
data from the right places in memory. 
I was a collaborator in the code that actually evaluates the one electron integrals,
however it hasn't been published and my collaborators were not keen on releasing
the code for the actual cuda kernels. This also applies for the two electron integrals. 
Refer to for details about performance of the two electron integrals:
https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00768
https://ieeexplore.ieee.org/abstract/document/9355281
Sorry for not being able to release the other code. 
*/ 
/**
 * @brief Compute one electron integral routines
 * @author Jorge Galvez
 *
 * @param shp_bin
 * @param n_atoms
 * @param R
 * @param Z
 * @param shells
 */
void DeviceKit::compute_onee(genfock::GPU_BraBatch_Bin &shp_bin,
                             const unsigned int n_atoms, double *R, double *Z,
                             Basis &shells) {

  // need to loop over it so I can access the batches
  unsigned n_shp_types = shp_bin.num_shp_types();

  for (unsigned int ab_type_idx = 0; ab_type_idx < n_shp_types; ++ab_type_idx) {

    for (unsigned int ab_size_idx = 0; ab_size_idx < 10; ++ab_size_idx) {

      for (unsigned int ab_batch_idx = 0;
           ab_batch_idx < shp_bin.container()[ab_type_idx][ab_size_idx].size();
           ++ab_batch_idx) {
        unsigned nab = shp_bin.sizes()[ab_type_idx][ab_size_idx][ab_batch_idx];
        m_bra_batch =
            &(shp_bin.container()[ab_type_idx][ab_size_idx][ab_batch_idx]);
        unsigned nppab = m_bra_batch->zeta_vec.size();
        m_bra_batch->create_array_maps();

        unsigned n_executing_devices(1);

        unsigned am_a = m_bra_batch->am_a;
        unsigned am_b = m_bra_batch->am_b;

        std::vector<size_t> nab_v(n_executing_devices); // should be m_n_devices
        std::vector<size_t> nppab_v(n_executing_devices);
        std::vector<size_t> nandb_v(n_executing_devices);
        std::vector<size_t> nppab3_v(n_executing_devices);
        std::vector<size_t> na_v(n_executing_devices);
        std::vector<size_t> nb_v(n_executing_devices);
        std::vector<double *> Ax_v(n_executing_devices);
        std::vector<double *> Ay_v(n_executing_devices);
        std::vector<double *> Az_v(n_executing_devices);
        std::vector<double *> ABx_v(n_executing_devices);
        std::vector<double *> ABy_v(n_executing_devices);
        std::vector<double *> ABz_v(n_executing_devices);
        std::vector<double *> UP_v(n_executing_devices);
        std::vector<double *> P_v(n_executing_devices);
        std::vector<double *> zeta_v(n_executing_devices);
        std::vector<double *> beta_v(n_executing_devices);
        std::vector<double *> fz_v(n_executing_devices);
        std::vector<double *> R_v(n_executing_devices);
        std::vector<double *> Z_v(n_executing_devices);
        std::vector<unsigned *> Kab_v(n_executing_devices);
        std::vector<unsigned *> offab_v(n_executing_devices);
        std::vector<unsigned *> offa_v(n_executing_devices);
        std::vector<unsigned *> offb_v(n_executing_devices);

#pragma omp parallel for schedule(static)
        for (unsigned idevice = 0; idevice < n_executing_devices; ++idevice) {

          // prepare ab info
          nab_v[idevice] = nab;
          nppab_v[idevice] = nppab;
          nppab3_v[idevice] = 3 * nppab;
          na_v[idevice] = m_bra_batch->n_a;
          nb_v[idevice] = m_bra_batch->n_b;
          Ax_v[idevice] = m_bra_batch->Ax_vec.data();
          Ay_v[idevice] = m_bra_batch->Ay_vec.data();
          Az_v[idevice] = m_bra_batch->Az_vec.data();
          ABx_v[idevice] = m_bra_batch->ABx_vec.data();
          ABy_v[idevice] = m_bra_batch->ABy_vec.data();
          ABz_v[idevice] = m_bra_batch->ABz_vec.data();
          UP_v[idevice] = m_bra_batch->UP_vec.data();
          P_v[idevice] = m_bra_batch->P_vec.data();
          zeta_v[idevice] = m_bra_batch->zeta_vec.data();
          beta_v[idevice] = m_bra_batch->beta_vec.data();
          fz_v[idevice] = m_bra_batch->fz_vec.data();
          Kab_v[idevice] = m_bra_batch->Kab.data();
          offab_v[idevice] = m_bra_batch->ppair_ab_offsets.data();
          offa_v[idevice] = m_bra_batch->offsets_sha.data();
          offb_v[idevice] = m_bra_batch->offsets_shb.data();
          R_v[idevice] = R;
          Z_v[idevice] = Z;

        } // for

        // now got to do the host pinned mem
        std::vector<double *> m_double_host_pinned_beta;
        std::vector<double *> m_double_host_pinned_R;
        std::vector<double *> m_int_host_pinned_Z;
        std::vector<double *> m_double_device_beta;
        std::vector<double *> m_double_device_R;
        std::vector<double *> m_int_device_Z;

        m_double_host_pinned_beta.resize(m_n_devices);
        m_double_host_pinned_R.resize(m_n_devices);
        m_int_host_pinned_Z.resize(m_n_devices);
        m_double_device_beta.resize(m_n_devices);
        m_double_device_R.resize(m_n_devices);
        m_int_device_Z.resize(m_n_devices);

        for (int idevice = 0; idevice < m_n_devices; ++idevice) {
          cudaSetDevice(m_device_id + idevice);
          gpuAssert(
              cudaMallocHost((void **)&(m_double_host_pinned_beta[idevice]),
                             nppab_v[idevice] * sizeof(double)));
          cudaMallocHost((void **)&(m_double_host_pinned_R[idevice]),
                         (n_atoms * 3) * sizeof(double));
          cudaMallocHost((void **)&(m_int_host_pinned_Z[idevice]),
                         (n_atoms) * sizeof(double));

          gpuAssert(cudaMalloc((void **)&(m_double_device_beta[idevice]),
                               nppab_v[idevice] * sizeof(double)));
          cudaMalloc((void **)&(m_double_device_R[idevice]),
                     (n_atoms * 3) * sizeof(double));
          cudaMalloc((void **)&(m_int_device_Z[idevice]),
                     (n_atoms) * sizeof(double));

          double *ihost_ptr_beta = (m_double_host_pinned_beta[idevice]);
          double *ihost_ptr_R = (m_double_host_pinned_R[idevice]);
          double *ihost_ptr_Z = (m_int_host_pinned_Z[idevice]);

          std::memcpy(ihost_ptr_beta, beta_v[idevice],
                      nppab_v[idevice] * sizeof(double));
          std::memcpy(ihost_ptr_R, R_v[idevice],
                      (n_atoms * 3) * sizeof(double));
          std::memcpy(ihost_ptr_Z, Z_v[idevice], n_atoms * sizeof(double));

          // cudaStreamSynchronize(m_streams[idevice]);
        }

        m_mem_pool.host_memcpy_shp_ab(Ax_v, nab_v, "Ax", n_executing_devices);
        m_mem_pool.host_memcpy_shp_ab(Ay_v, nab_v, "Ay", n_executing_devices);
        m_mem_pool.host_memcpy_shp_ab(Az_v, nab_v, "Az", n_executing_devices);
        m_mem_pool.host_memcpy_shp_ab(ABx_v, nab_v, "ABx", n_executing_devices);
        m_mem_pool.host_memcpy_shp_ab(ABy_v, nab_v, "ABy", n_executing_devices);
        m_mem_pool.host_memcpy_shp_ab(ABz_v, nab_v, "ABz", n_executing_devices);

        m_mem_pool.host_memcpy_pp_ab(UP_v, nppab_v, "UP", n_executing_devices);
        m_mem_pool.host_memcpy_pp_ab(P_v, nppab3_v, "P", n_executing_devices);
        m_mem_pool.host_memcpy_pp_ab(fz_v, nppab_v, "fz", n_executing_devices);
        m_mem_pool.host_memcpy_pp_ab(zeta_v, nppab_v, "zeta",
                                     n_executing_devices);

        m_mem_pool.host_memcpy_shp_ab_uint(Kab_v, nab_v, "Kab",
                                           n_executing_devices);

        m_mem_pool.host_memcpy_shp_ab_uint(offab_v, nab_v, "offab",
                                           n_executing_devices);

        m_mem_pool.host_memcpy_shp_ab_uint(offa_v, nab_v, "offa",
                                           n_executing_devices);

        m_mem_pool.host_memcpy_shp_ab_uint(offb_v, nab_v, "offb",
                                           n_executing_devices);

        for (int idevice = 0; idevice < m_n_devices; ++idevice) {

          cudaSetDevice(m_device_id + idevice);

          gpuAssert(cudaMemcpyAsync(
              m_double_device_beta[idevice], m_double_host_pinned_beta[idevice],
              nppab_v[idevice] * sizeof(double), cudaMemcpyHostToDevice,
              m_streams[idevice]));

          cudaMemcpyAsync(m_double_device_R[idevice],
                          m_double_host_pinned_R[idevice],
                          (n_atoms * 3) * sizeof(double),
                          cudaMemcpyHostToDevice, m_streams[idevice]);
          cudaMemcpyAsync(m_int_device_Z[idevice], m_int_host_pinned_Z[idevice],
                          (n_atoms) * sizeof(double), cudaMemcpyHostToDevice,
                          m_streams[idevice]);

          // cudaStreamSynchronize(m_streams[idevice]);
        }

        m_mem_pool.gpu_memcpy_shp_ab("Ax", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab("Ay", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab("Az", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab("ABx", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab("ABy", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab("ABz", m_streams, n_executing_devices);

        m_mem_pool.gpu_memcpy_pp_ab("UP", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_pp_ab("P", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_pp_ab("fz", m_streams, n_executing_devices);
        m_mem_pool.gpu_memcpy_pp_ab("zeta", m_streams, n_executing_devices);

        m_mem_pool.gpu_memcpy_shp_ab_uint("Kab", m_streams,
                                          n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab_uint("offab", m_streams,
                                          n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab_uint("offa", m_streams,
                                          n_executing_devices);
        m_mem_pool.gpu_memcpy_shp_ab_uint("offb", m_streams,
                                          n_executing_devices);
        // call scheduler
        for (int idevice = 0; idevice < m_n_devices; ++idevice) {

          cudaSetDevice(m_device_id + idevice);
          cudaStreamSynchronize(m_streams[idevice]);
          Kinetic_Overlap_scheduler(
              m_mem_pool.get_device_shp_uint_ptr("Kab", idevice),
              m_mem_pool.get_device_shp_ptr("Ax", idevice),
              m_mem_pool.get_device_shp_ptr("Ay", idevice),
              m_mem_pool.get_device_shp_ptr("Az", idevice),
              m_mem_pool.get_device_shp_ptr("ABx", idevice),
              m_mem_pool.get_device_shp_ptr("ABy", idevice),
              m_mem_pool.get_device_shp_ptr("ABz", idevice),
              m_mem_pool.get_device_pp_ptr("P", idevice),
              m_mem_pool.get_device_pp_ptr("zeta", idevice),
              m_double_device_beta[idevice],
              m_mem_pool.get_device_pp_ptr("UP", idevice),
              m_mem_pool.get_device_pp_ptr("fz", idevice),
              m_mem_pool.get_device_shp_uint_ptr("offa", idevice),
              m_mem_pool.get_device_shp_uint_ptr("offb", idevice),
              m_bra_batch->batch_size, m_n_bas_functs,
              this->get_Hcore_matrix_ptr(), this->get_S_matrix_ptr(), 128,
              m_streams[idevice], am_a, am_b);

          nuc_attr_scheduler(
              m_mem_pool.get_device_shp_uint_ptr("Kab", idevice),
              m_mem_pool.get_device_shp_ptr("Ax", idevice),
              m_mem_pool.get_device_shp_ptr("Ay", idevice),
              m_mem_pool.get_device_shp_ptr("Az", idevice),
              m_mem_pool.get_device_shp_ptr("ABx", idevice),
              m_mem_pool.get_device_shp_ptr("ABy", idevice),
              m_mem_pool.get_device_shp_ptr("ABz", idevice),
              m_mem_pool.get_device_pp_ptr("P", idevice),
              m_double_device_R[idevice],
              m_mem_pool.get_device_pp_ptr("zeta", idevice), n_atoms,
              m_int_device_Z[idevice],
              m_mem_pool.get_device_pp_ptr("UP", idevice),
              m_mem_pool.get_device_pp_ptr("fz", idevice),
              m_mem_pool.get_device_shp_uint_ptr("offa", idevice),
              m_mem_pool.get_device_shp_uint_ptr("offb", idevice),
              m_bra_batch->batch_size, m_n_bas_functs,
              this->get_Hcore_matrix_ptr(), 128, m_streams[idevice], am_a,
              am_b);
          cudaStreamSynchronize(m_streams[idevice]);

        } // for scheduler
          // free memory
        for (int idevice = 0; idevice < m_n_devices; ++idevice) {
          cudaSetDevice(m_device_id + idevice);
          gpuAssert(cudaFreeHost(m_double_host_pinned_beta[idevice]));
          gpuAssert(cudaFreeHost(m_double_host_pinned_R[idevice]));
          gpuAssert(cudaFreeHost(m_int_host_pinned_Z[idevice]));

          gpuAssert(cudaFree(m_double_device_beta[idevice]));
          gpuAssert(cudaFree(m_double_device_R[idevice]));
          gpuAssert(cudaFree(m_int_device_Z[idevice]));
        }

      } // batches
    }   // sizes
  }     // types
  correct_overlap(shells);
  correct_hcore(shells);
} // compute_onee
