
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/Globdat.h>
#include <jive/algebra/VectorSpace.h>


JEM_DEFINE_CLASS( jive::algebra::VectorSpace );


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   class VectorSpace
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


VectorSpace::~VectorSpace ()
{}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t VectorSpace::size () const
{
  return getDofSpace()->dofCount ();
}


//-----------------------------------------------------------------------
//   isDistributed
//-----------------------------------------------------------------------


bool VectorSpace::isDistributed () const
{
  return ( globalSize() > size() );
}


//-----------------------------------------------------------------------
//   norm2
//-----------------------------------------------------------------------


double VectorSpace::norm2 ( const Vector& x ) const
{
  return std::sqrt ( product( x, x ) );
}


//-----------------------------------------------------------------------
//   products
//-----------------------------------------------------------------------


void VectorSpace::products

  ( const Vector&  a,
    const Matrix&  x,
    const Matrix&  y ) const

{
  JEM_PRECHECK2 ( x.size(1) == a.size () &&
                  x.size(1) == y.size(1),
                  "Array shape mismatch" );

  const idx_t  n = y.size (1);

  for ( idx_t i = 0; i < n; i++ )
  {
    a[i] = product ( x[i], y[i] );
  }
}


//-----------------------------------------------------------------------
//   project
//-----------------------------------------------------------------------


void VectorSpace::project

  ( const Vector&  a,
    const Vector&  x,
    const Matrix&  y ) const

{
  JEM_PRECHECK2 ( a.size() >= y.size(1),
                  "coefficient array is too small" );

  const idx_t  n = y.size (1);

  for ( idx_t i = 0; i < n; i++ )
  {
    a[i] = product ( x, y[i] );
  }

  if ( a.size() > n )
  {
    a[n] = product ( x, x );
  }
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void VectorSpace::store

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat ) const

{
  Ref<Self> self ( const_cast<Self*>( this ) );

  Globdat::storeFor ( Globdat::VECTOR_SPACE, self, dofs, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<VectorSpace> VectorSpace::find

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<Self>  vspace;

  Globdat::findFor ( vspace, Globdat::VECTOR_SPACE, dofs, globdat );

  return vspace;
}


JIVE_END_PACKAGE( algebra )
