/*
 *  Isotropic Material Header
 *  
 *  Isotropic Material Constitutive Relations
 *
 *  Ozgur Taylan Turan, February 2022
 *
 */

#ifndef ISOTROPIC_MATERIAL_H
#define ISOTROPIC_MATERIAL_H

//------------------------------------------------------------------------------
//    Additional Headers
//------------------------------------------------------------------------------


#include "Material.h"


//------------------------------------------------------------------------------
//    class IsotropicMaterial
//------------------------------------------------------------------------------


class IsotropicMaterial: public Material
{
  public:

  static const char*     E_PROP;
  static const char*     NU_PROP;
  static const char*     ASSUMPTION_PROP;
  static const char*     AREA_PROP;

  IsotropicMaterial
    ( const idx_t       rank,
      const Properties& globdat );

  virtual void configure
    ( const Properties& props,
      const Properties& globdat );

  virtual void getConfig
    ( const Properties& conf,
      const Properties& globdat ) const;

  virtual void update
    ( Matrix&       stiff,
      Vector&       stress,
      const Vector& strain,
      const idx_t   ipoint ) = 0;

  virtual void addTableColumns
    ( IdxVector&    jcols,
      XTable&       table,
      const String& name ) = 0;

  virtual Ref<Material> clone ( ) const = 0;

  protected:

  ~IsotropicMaterial ();

  private:

  idx_t                   rank_;
  double                  E_;
  double                  nu_;
  double                  A_;
  Matrix                  stiffMatrix_;
  String                  assumption_;
  
  void computeStiffMatrix_ ();

};


#endif
