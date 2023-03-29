
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/geom/SimpleMaskMatrix.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::SimpleMaskMatrix );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class SimpleMaskMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SimpleMaskMatrix::SimpleMaskMatrix ()
{}


SimpleMaskMatrix::SimpleMaskMatrix ( const BoolMatrix& mask ) :

  mask_ ( mask )

{
  JEM_PRECHECK2 ( mask.size(0) == mask.size(1),
                  "non-square mask matrix" );
}


SimpleMaskMatrix::~SimpleMaskMatrix ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void SimpleMaskMatrix::readFrom ( ObjectInput&  in )
{
  decode ( in, mask_ );
}


void SimpleMaskMatrix::writeTo  ( ObjectOutput& out ) const
{
  encode ( out, mask_ );
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


bool SimpleMaskMatrix::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  return mask_(irow,jcol);
}


//-----------------------------------------------------------------------
//   getRowValues
//-----------------------------------------------------------------------


void SimpleMaskMatrix::getRowValues

  ( const BoolVector&  values,
    idx_t              irow,
    const IdxVector&   jcols ) const

{
  JEM_PRECHECK2 ( values.size() == jcols.size(),
                  "Array size mismatch" );

  const idx_t  n = values.size ();

  for ( idx_t j = 0; j < n; j++ )
  {
    values[j] = mask_(irow,jcols[j]);
  }
}


JIVE_END_PACKAGE( geom )
