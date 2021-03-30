/**
 * @file sad.cpp
 * @author Jorge Galvez (gbarca@gbarca.com)
 * @brief Contains the SAD guess routine
 * @version 0.1
 * @date 2021-01-12
 * 
 * @copyright Copyright (c) 2021
 * 
 */
/**
This routine creates the SAD initial guess for the SCF process. It is 
actually quite simple, it reads precalculated densities from an hdf5 file 
and creates the initial guess density matrix. 
*/
#include "../basis/bsed_mappings.hpp"
#include "sad.hpp"
#include <H5Cpp.h>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <time.h>
#include <vector>

namespace genfock {
/**
 * @brief SAD guess driver routine. Reads in from HDF5 file.
 * @author Jorge Galvez
 * @date December 2020
 * @param molecule Molecule object
 * @param nbas Number of basis functions
 * @param basis Basis function string
 * @param sad_guess_array Vector for sad guess 
 */
void sad_guess(Molecule &molecule, unsigned int nbas, std::string basis,
               std::vector<double> &sad_guess_array) {

  std::vector<unsigned int> atom_list;
  std::vector<unsigned int> nbas_list;
  std::vector<unsigned int> nguess_list;
  std::vector<double> sad_vector;
  BasisSetMappings mappings;

  for (int iatom = 0; iatom != molecule.natoms(); ++iatom) {
    // push back the atom number into the list
    // this is needed for the guess reader
    atom_list.push_back(mappings.atom_to_atomic_number(molecule.atoms(iatom)));

  } // for iatom

  // let's get the sad guess file
  H5::H5File file("records/sadgss.h5", H5F_ACC_RDONLY);
  for (int iatom = 0; iatom < atom_list.size(); ++iatom) {
    // get the atom symbol
    std::string atom = mappings.atomic_number_to_atom(atom_list[iatom]);
    // use the atom symbol and read-in basis to form the pair
    std::string atom_basis_pair = atom + "/" + basis;
    H5::DataSet guess_dataset = file.openDataSet(atom_basis_pair);
    H5::DataSpace guess_dataspace = guess_dataset.getSpace();

    // the guess num elements of each atom tell us how many basis functions are
    // there
    hsize_t guess_num_elements;
    int guess_rank =
        guess_dataspace.getSimpleExtentDims(&guess_num_elements, NULL);

    // get basis functions and guess sizes for a given atom and store them
    int nbasis_element = int(sqrt(guess_num_elements));
    nguess_list.push_back(int(guess_num_elements));
    nbas_list.push_back(int(nbasis_element));

    //-- read from data set --//
    std::vector<double> guess_buf(guess_num_elements, 0.0);
    guess_dataset.read(guess_buf.data(), H5::PredType::NATIVE_DOUBLE);

    // the sad vector only contains the densities, we need 0s in the sad guess
    sad_vector.insert(sad_vector.end(), guess_buf.begin(), guess_buf.end());

  } // iatom

  // now we have to loop over the atoms and write the densities in block
  // diagonal form!
  int index = 0; // gives where to read from the sad_vector which only contains
                 // densities!
  int sad_offset = 0; // controls where to write depending on the size of the
                      // sad matrix of the atom
  int level = 0;      // where in the matrix are we writing

  for (int i = 0; i < atom_list.size(); ++i) {
    for (int j = sad_offset; j < (sad_offset + nbas_list[i]); ++j, ++level) {
      for (int k = 0, l = index; k < nbas_list[i]; ++k, ++l) {
        sad_guess_array[sad_offset + (level * nbas) + k] = sad_vector[l];
      } // k
      index += nbas_list[i];
    } // j
    sad_offset += nbas_list[i];
  } // i
} // sad_guess

/**
 * @brief Optional printing of the full sad guess to console output
 * 
 * @author Jorge Galvez
 * @date December 2020
 * @param nbas Number of basis functions
 * @param basis Basis string
 * @param sad_guess_array Sad array
 */
void print_sad_guess(unsigned int nbas, std::string basis,
                     std::vector<double> &sad_guess_array) {

  std::cerr << " Printing the sad guess for " << basis << std::endl;
  // for debug, print the ENTIRE sad guess
  for (int i = 0; i < nbas; ++i) {
    for (int j = 0; j < nbas; ++j) {
      std::cerr << i << ", " << j << " => " << sad_guess_array[nbas * i + j]
                << std::endl;
    } // for j
  }   // for i

  /*
  H5::H5File file("debug.h5", H5F_ACC_TRUNC);

  // dataset dimensions
  hsize_t dimsf[1];
  dimsf[0] = sad_guess_array.size();
  H5::DataSpace dataspace(1, dimsf);

  H5::DataType datatype(H5::PredType::NATIVE_DOUBLE);
  H5::DataSet dataset = file.createDataSet("sad_guess", datatype, dataspace);

  // write
  dataset.write(sad_guess_array.data(), H5::PredType::NATIVE_DOUBLE);
  */
}
} // namespace genfock
