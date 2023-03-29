
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
#include <jem/base/array/operators.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/GramSchmidtFactory.h>
#include <jive/algebra/ClassicalGramSchmidt.h>


JEM_DEFINE_CLASS( jive::algebra::ClassicalGramSchmidt );


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   class ClassicalGramSchmidt
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ClassicalGramSchmidt::TYPE_NAME = "CGS";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ClassicalGramSchmidt::ClassicalGramSchmidt

  ( const String&     name,
    Ref<VectorSpace>  vspace ) :

    Super ( name )

{
  JEM_PRECHECK2 ( vspace, "NULL VectorSpace" );

  vspace_ = vspace;
}


ClassicalGramSchmidt::~ClassicalGramSchmidt ()
{}


//-----------------------------------------------------------------------
//   orthogonalize
//-----------------------------------------------------------------------


void ClassicalGramSchmidt::orthogonalize

  ( const Vector&  a,
    const Vector&  x,
    const Matrix&  y ) const

{
  JEM_PRECHECK2 ( a.size() >= y.size(1),
                  "coefficient array is too small" );

  const idx_t  m = jem::min ( a.size(), y.size(1) + 1 );
  const idx_t  n = y.size(1);


  vspace_->project ( a[slice(BEGIN,m)], x, y );

  x = x - jem::numeric::matmul ( y, a[slice(BEGIN,n)] );

  if ( a.size() > n )
  {
    a[n] = std::sqrt ( a[n] );
  }
  if ( a.size() > n + 1 )
  {
    a[n + 1] = vspace_->norm2 ( x );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<GramSchmidt> ClassicalGramSchmidt::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    Ref<VectorSpace>   vspace )

{
  return jem::newInstance<Self> ( name, vspace );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ClassicalGramSchmidt::declare ()
{
  GramSchmidtFactory::declare ( TYPE_NAME,  & makeNew );
  GramSchmidtFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( algebra )
