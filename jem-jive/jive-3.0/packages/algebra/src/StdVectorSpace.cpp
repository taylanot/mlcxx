
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/StdVectorSpace.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::StdVectorSpace );


JIVE_BEGIN_PACKAGE( algebra )


using jem::dot;


//=======================================================================
//   class StdVectorSpace
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdVectorSpace::StdVectorSpace ()
{}


StdVectorSpace::StdVectorSpace ( const Ref<DofSpace>& dofs ) :

  dofs_ ( dofs )

{
  JEM_PRECHECK2 ( dofs, "NULL DofSpace" );
}


StdVectorSpace::~StdVectorSpace ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdVectorSpace::readFrom ( ObjectInput& in )
{
  dofs_ = jem::checkedCast<DofSpace> ( in.decodeObject() );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdVectorSpace::writeTo ( ObjectOutput& out ) const
{
  out.encodeObject ( dofs_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdVectorSpace::clone () const
{
  return jem::newInstance<Self> ( dofs_ );
}


//-----------------------------------------------------------------------
//   globalSize
//-----------------------------------------------------------------------


idx_t StdVectorSpace::globalSize () const
{
  return dofs_->dofCount ();
}


//-----------------------------------------------------------------------
//   norm2
//-----------------------------------------------------------------------


double StdVectorSpace::norm2 ( const Vector& x ) const
{
  JEM_ASSERT2 ( x.size() == this->size(), "Array has wrong size" );

  return jem::numeric::norm2 ( x );
}


//-----------------------------------------------------------------------
//   product
//-----------------------------------------------------------------------


double StdVectorSpace::product

  ( const Vector&  x,
    const Vector&  y ) const

{
  JEM_ASSERT2   ( x.size() == this->size(), "Array has wrong size" );
  JEM_PRECHECK2 ( x.size() == y    .size(), "Array size mismatch" );

  return dot ( x, y );
}


//-----------------------------------------------------------------------
//   products
//-----------------------------------------------------------------------


void StdVectorSpace::products

  ( const Vector&  a,
    const Matrix&  x,
    const Matrix&  y ) const

{
  JEM_ASSERT2   ( x.size(0) == this->size(), "Array has wrong shape" );

  JEM_PRECHECK2 ( a.size()  == x.size(1) &&
                  x.size(0) == y.size(0) &&
                  x.size(1) == y.size(1),
                  "Array size mismatch" );

  const idx_t  n = y.size (1);

  for ( idx_t i = 0; i < n; i++ )
  {
    a[i] = dot ( x[i], y[i] );
  }
}


//-----------------------------------------------------------------------
//   project
//-----------------------------------------------------------------------


void StdVectorSpace::project

  ( const Vector&  a,
    const Vector&  x,
    const Matrix&  y ) const

{
  using jem::numeric::matmul;

  JEM_ASSERT2   ( x.size() == this->size(), "Array has wrong size" );
  JEM_PRECHECK2 ( a.size() >= y.size(1) &&
                  x.size() == y.size(0),
                  "Array size mismatch" );

  const idx_t  n = y.size (1);

  if      ( x.size() == 0 )
  {
    a = 0.0;
  }
  else if ( a.size() == n )
  {
    matmul ( a, x, y );
  }
  else
  {
    matmul ( a[slice(BEGIN,n)], x, y );

    a[n] = dot ( x, x );
  }
}


//-----------------------------------------------------------------------
//   getDofSpace
//-----------------------------------------------------------------------


DofSpace* StdVectorSpace::getDofSpace () const
{
  return dofs_.get ();
}


JIVE_END_PACKAGE( algebra )
