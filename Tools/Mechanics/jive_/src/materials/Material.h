/*
 *  Material Header
 *  
 *  Ozgur Taylan Turan, February 2022
 *
 */

#ifndef MATERIAL_H
#define MATERIAL_H

//------------------------------------------------------------------------------
//    Base Headers
//------------------------------------------------------------------------------


#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/util/XTable.h>


//------------------------------------------------------------------------------
//    Using Declarations
//------------------------------------------------------------------------------


using jem::System;
using jem::idx_t;
using jem::Ref;
using jem::Object;
using jem::String;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::util::XTable;


//------------------------------------------------------------------------------
//    class Material
//------------------------------------------------------------------------------


class Material : public Object
{
  public:

  Material
    ( const idx_t       rank,
      const Properties& globdat );

  Material
    ( const String&     name,
      const Properties& props,
      const Properties& conf,
      const idx_t       rank );

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

  virtual void stressAtPoint
    ( Vector&       stress,
      const Vector& strain,
      const idx_t   ipoint ) = 0;

  virtual void stiffAtPoint
    ( Vector&     stiffvec,
      const idx_t ipoint ) = 0;

  virtual void addTableColumns
    ( IdxVector&    jcols,
      XTable&       table,
      const String& name ) = 0;

  virtual void getHistory
    ( Vector&     hvals,
      const idx_t mpoint );

  virtual void setHistory
    ( const Vector& hvals,
      const idx_t   mpoint );

  virtual void createIntPoints
    ( const idx_t npoints );

  virtual void commit ();
  
  virtual double getDissipation
    ( const idx_t ipoint ) const;

  virtual double getDissipationGrad
    ( const idx_t ipoint ) const;

  virtual idx_t pointCount () const;

  virtual bool isLoading 
    ( const idx_t ipoint ) const;

  virtual bool wasLoading
    ( const idx_t ipoint ) const;

  virtual bool isInelastic
    ( const idx_t ipoint ) const;

  virtual void writeState ();

  virtual Ref<Material> clone ( ) const = 0;

 protected:

  ~Material ();

};


//------------------------------------------------------------------------------
//    newMaterial
//------------------------------------------------------------------------------


Ref<Material>  newMaterial
    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );


#endif
