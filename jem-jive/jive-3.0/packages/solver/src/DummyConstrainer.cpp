
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
#include <jive/solver/ConstrainerFactory.h>
#include <jive/solver/DummyConstrainer.h>


JEM_DEFINE_CLASS( jive::solver::DummyConstrainer );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class DummyConstrainer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DummyConstrainer::TYPE_NAME = "None";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DummyConstrainer::DummyConstrainer

  ( const String&        name,
    Ref<AbstractMatrix>  matrix ) :

    Super   (   name ),
    matrix_ ( matrix )

{
  JEM_PRECHECK ( matrix );
}


DummyConstrainer::~DummyConstrainer ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void DummyConstrainer::update ()
{}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


bool DummyConstrainer::updated () const
{
  return true;
}


//-----------------------------------------------------------------------
//   getRhs
//-----------------------------------------------------------------------


void DummyConstrainer::getRhs

  ( const Vector&  g,
    const Vector&  f ) const

{
  g = f;
}


//-----------------------------------------------------------------------
//   getLhs
//-----------------------------------------------------------------------


void DummyConstrainer::getLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  x = y;
}


//-----------------------------------------------------------------------
//   initLhs
//-----------------------------------------------------------------------


void DummyConstrainer::initLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  x = y;
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* DummyConstrainer::getConstraints () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getInputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* DummyConstrainer::getInputMatrix () const
{
  return matrix_.get ();
}


//-----------------------------------------------------------------------
//   getOutputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* DummyConstrainer::getOutputMatrix () const
{
  return matrix_.get ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Constrainer> DummyConstrainer::makeNew

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix )

{
  return jem::newInstance<Self> ( name, matrix );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DummyConstrainer::declare ()
{
  ConstrainerFactory::declare ( TYPE_NAME,  & makeNew );
  ConstrainerFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( solver )
