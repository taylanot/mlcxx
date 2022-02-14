/*
 *  Isotropic Material Header
 *  
 *  Isotropic Material Constitutive Relations
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */
//------------------------------------------------------------------------------
//    Base Headers
//------------------------------------------------------------------------------


#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>


//------------------------------------------------------------------------------
//    Additional Headers
//------------------------------------------------------------------------------


#include "IsotropicMaterial.h"


//------------------------------------------------------------------------------
//    Using Declarations 
//------------------------------------------------------------------------------


using namespace jem;
using jem::numeric::matmul;


//==============================================================================
//    class IsotropicMaterial
//==============================================================================


//------------------------------------------------------------------------------
//    static data
//------------------------------------------------------------------------------


const char* IsotropicMaterial::E_PROP           = "E";
const char* IsotropicMaterial::NU_PROP          = "nu";
const char* IsotropicMaterial::AREA_PROP        = "area";
const char* IsotropicMaterial::ASSUMPTION_PROP  = "anmodel";


//------------------------------------------------------------------------------
//    constructors & destrcutor
//------------------------------------------------------------------------------


IsotropicMaterial::IsotropicMaterial
  ( const idx_t       rank,
    const Properties& globdat ) : Material ( rank, globdat )
{
  rank_ = rank;

  // Try to understand why you would create STRAIN_COUNTS[4] with zero included
  const idx_t STRAIN_COUNTS[3] = {1, 3, 6};

  JEM_PRECHECK ( rank_ >= 1 && rank_ <= 3);

  E_  = 1.0;
  nu_ = 0.0;
  A_  = 1.0;
  
  stiffMatrix_.resize ( STRAIN_COUNTS[rank_], STRAIN_COUNTS[rank_] );
  stiffMatrix_ = 0.;
}

IsotropicMaterial::~IsotropicMaterial ()
{}

