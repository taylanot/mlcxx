
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
#include <jive/algebra/NullMatrixObject.h>
#include <jive/algebra/NullMatrixBuilder.h>


JEM_DEFINE_CLASS( jive::algebra::NullMatrixBuilder );


JIVE_BEGIN_PACKAGE( algebra )


using jem::newInstance;


//=======================================================================
//   class NullMatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NullMatrixBuilder::TYPE_NAME = "Null";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NullMatrixBuilder::NullMatrixBuilder ( const String& name ) :

  Super ( name )

{
  matrix_     = newInstance<NullMatrixObj> ();
  multiplier_ = 1.0;
}


NullMatrixBuilder::~NullMatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void NullMatrixBuilder::clear ()
{}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void NullMatrixBuilder::scale ( double s )
{}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void NullMatrixBuilder::setToZero ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void NullMatrixBuilder::reserve ( idx_t n )
{}


//-----------------------------------------------------------------------
//   shapeHint
//-----------------------------------------------------------------------


void NullMatrixBuilder::shapeHint

  ( idx_t  rowCount,
    idx_t  colCount )

{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void NullMatrixBuilder::trimToSize ()
{}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void NullMatrixBuilder::updateMatrix ()
{}


//-----------------------------------------------------------------------
//   setMultiplier
//-----------------------------------------------------------------------


void NullMatrixBuilder::setMultiplier ( double x )
{
  multiplier_ = x;
}


//-----------------------------------------------------------------------
//   getMultiplier
//-----------------------------------------------------------------------


double NullMatrixBuilder::getMultiplier () const
{
  return multiplier_;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void NullMatrixBuilder::setData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  data )

{}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void NullMatrixBuilder::addData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  data )

{}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t NullMatrixBuilder::eraseData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount )

{
  return 0;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void NullMatrixBuilder::getData

  ( double*       buf,
    const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount ) const

{
  const idx_t  n = icount * jcount;

  for ( idx_t i = 0; i < n; i++ )
  {
    buf[i] = 0.0;
  }
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* NullMatrixBuilder::getMatrix () const
{
  return matrix_.get ();
}


JIVE_END_PACKAGE( algebra )
