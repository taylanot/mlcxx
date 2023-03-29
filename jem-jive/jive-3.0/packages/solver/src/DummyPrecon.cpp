
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
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/DummyPrecon.h>


JEM_DEFINE_CLASS( jive::solver::DummyPrecon );


using jive::algebra::MatrixTraits;


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class DummyPrecon
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DummyPrecon::TYPE_NAME = "None";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DummyPrecon::DummyPrecon

  ( const String&        name,
    Ref<AbstractMatrix>  matrix ) :

    Super ( name ),
    size_ (    0 )

{
  if ( matrix )
  {
    Shape sh ( matrix->shape() );

    if ( sh[0] != sh[1] )
    {
      util::nonSquareMatrixError ( getContext(), sh );
    }

    size_ = sh[0];

    jem::util::connect ( matrix->newStructEvent,
                         this, & Self::resize_ );
  }
}


DummyPrecon::~DummyPrecon ()
{}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape DummyPrecon::shape () const
{
  return Shape ( size_, size_ );
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void DummyPrecon::start ()
{}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void DummyPrecon::finish ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void DummyPrecon::update ()
{}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void DummyPrecon::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  lhs = rhs;
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool DummyPrecon::hasTrait ( const String& trait ) const
{
  if ( trait == MatrixTraits::SYMMETRIC )
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void DummyPrecon::resize ( idx_t n )
{
  JEM_PRECHECK ( n >= 0 );

  idx_t  sz = size_;

  size_ = n;

  if ( sz != n )
  {
    newStructEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Precon> DummyPrecon::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<AbstractMatrix>  mat;

  params.find ( mat, SolverParams::MATRIX );

  return jem::newInstance<Self> ( name, mat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DummyPrecon::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & makeNew );
  PreconFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   resize_
//-----------------------------------------------------------------------


void DummyPrecon::resize_ ( AbstractMatrix& mat )
{
  const Shape  sh = mat.shape ();

  if ( sh[0] != sh[1] )
  {
    util::nonSquareMatrixError ( CLASS_NAME, sh );
  }

  idx_t  sz = size_;

  size_  = sh[0];

  if ( sz != size_ )
  {
    newStructEvent.emit ( *this );
  }
}


JIVE_END_PACKAGE( solver )
