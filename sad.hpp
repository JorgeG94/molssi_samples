#ifndef SAD_HPP_
#define SAD_HPP_

/**
 * @file sad.hpp
 * @author Jorge Galvez (gbarca@gbarca.com)
 * @brief Sad header file
 * @version 0.1
 * @date 2021-01-12
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#include "../exess.hpp"
#include <string>
#include <vector>
namespace genfock {

/**
 * @brief Sad guess driver declaration. See sad.cpp
 * @author Jorge Galvez
 * @date December 2020
 * @param molecule Molecule object
 * @param nbas Number of basis functions
 * @param basis Basis set string
 * @param sad_guess_array Sad array
 */
void sad_guess(Molecule &molecule, unsigned int nbas, std::string basis,
               std::vector<double> &sad_guess_array);
/**
 * @brief Sad guess optional output declaration. See sad.cpp
 * @author Jorge Galvez
 * @date December 2020
 * @param nbas Number of basis functions
 * @param basis Basis set string
 * @param sad_guess_array Sad array
 */
void print_sad_guess(unsigned int nbas, std::string basis,
                     std::vector<double> &sad_guess_array);

} // namespace genfock
#endif
