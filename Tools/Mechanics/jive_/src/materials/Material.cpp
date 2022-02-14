/*
 *  Material Implementation
 *  
 *  Ozgur Taylan Turan, February 2022
 *
 */

//------------------------------------------------------------------------------
//    Base Headers
//------------------------------------------------------------------------------


#include <jem/base/Error.h>
#include <jem/util/Properties.h>


//------------------------------------------------------------------------------
//    Additional Headers
//------------------------------------------------------------------------------


#include "Material.h"
#include "IsotropicMaterial.h"

//------------------------------------------------------------------------------
//    Using Declarations
//------------------------------------------------------------------------------


using namespace jem;


//------------------------------------------------------------------------------
//    newInstance
//------------------------------------------------------------------------------


Ref<Material>  newMaterial
  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )
{
  Properties matProps = props.getProps ( name );
  Properties matConf  = conf.makeProps ( name );

  Ref<Material> mat;
  String        type;
  idx_t         rank;

  matProps.get ( type, "type" );
  matConf.set  ( "type", type );

  matProps.get ( rank, "rank" );
  matConf.set  ( "rank", rank );

  if ( type == "Isotropic" )
    mat = newInstance<IsotropicMaterial> ( rank, globdat );
  else
    matProps.propertyError ( name, "Invalid material: " + type );
  return mat;
}


//=======================================================================
//   class Material
//=======================================================================

//-----------------------------------------------------------------------
//   constructors and destructor
//-----------------------------------------------------------------------


Material::Material
  ( const idx_t        rank,
    const Properties&  globdat )
{}


Material::Material
  ( const String&      name,
    const Properties&  props,
    const Properties&  conf,
    const idx_t        rank )
{}


Material::~Material()
{}


//------------------------------------------------------------------------------
//    configure
//------------------------------------------------------------------------------


void Material::configure
  ( const Properties& props,
    const Properties& globdat )
{}


//------------------------------------------------------------------------------
//    getConfig
//------------------------------------------------------------------------------


void Material::getConfig

  ( const Properties& props,
    const Properties& globdat ) const
{}


//------------------------------------------------------------------------------
//    getHistory
//------------------------------------------------------------------------------


void Material::getHistory
  ( Vector&           hvals,
    const idx_t       mpoint ) 
{
  hvals.resize ( 0 );
}


//------------------------------------------------------------------------------
//   setHistory
//------------------------------------------------------------------------------


void Material::setHistory
  ( const Vector&    hvals,
    const idx_t      mpoint )

{}


//------------------------------------------------------------------------------
//    createIntPoints
//------------------------------------------------------------------------------


void Material::createIntPoints
  ( const idx_t       npoints )
{}


//------------------------------------------------------------------------------
//    commit 
//------------------------------------------------------------------------------


void Material::commit ()
{}


//------------------------------------------------------------------------------
//    getDissipation 
//------------------------------------------------------------------------------


double Material::getDissipation
  ( const idx_t      ipoint ) const
{
  return 0.;
}


//------------------------------------------------------------------------------
//    getDissipationGrad
//------------------------------------------------------------------------------


double Material::getDissipationGrad
  ( const idx_t      ipoint ) const
{
  return 0.;
}


//------------------------------------------------------------------------------
//    pointCount 
//------------------------------------------------------------------------------


idx_t  Material::pointCount () const

{
  return 0;
}


//------------------------------------------------------------------------------
//    isLoading 
//------------------------------------------------------------------------------


bool  Material::isLoading
  ( idx_t ipoint ) const

{
  return false;
}


//------------------------------------------------------------------------------
//    wasLoading 
//------------------------------------------------------------------------------


bool  Material::wasLoading 

  ( idx_t ipoint ) const

{
  return false;
}


//------------------------------------------------------------------------------
//    isInelastic
//------------------------------------------------------------------------------


bool Material::isInelastic
  ( idx_t ipoint ) const
{
  return false;
}


//------------------------------------------------------------------------------
//    writeState 
//------------------------------------------------------------------------------


void Material::writeState ()
{}

