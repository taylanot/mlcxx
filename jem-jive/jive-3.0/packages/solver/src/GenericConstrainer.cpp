
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jive/algebra/ConstrainedMatrix.h>
#include <jive/solver/ConstrainerFactory.h>
#include <jive/solver/GenericConstrainer.h>


JEM_DEFINE_CLASS( jive::solver::GenericConstrainer );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::System;


//=======================================================================
//   class GenericConstrainer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GenericConstrainer::TYPE_NAME = "Generic";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GenericConstrainer::GenericConstrainer

  ( const String&           name,
    Ref<ConstrainedMatrix>  matrix ) :

    Super   (   name ),
    output_ ( matrix )

{
  JEM_PRECHECK ( matrix );
}


GenericConstrainer::GenericConstrainer

  ( const String&        name,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix ) :

    Super ( name )

{
  JEM_PRECHECK ( cons && matrix );

  output_ = newInstance<ConstrainedMatrix> ( matrix, cons );
}


GenericConstrainer::~GenericConstrainer ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void GenericConstrainer::update ()
{
  if ( ! output_->updated() )
  {
    print ( System::debug( myName_ ), myName_,
            " : updating constrained matrix ...\n" );

    output_->update ();
  }
}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


bool GenericConstrainer::updated () const
{
  return output_->updated ();
}


//-----------------------------------------------------------------------
//   getRhs
//-----------------------------------------------------------------------


void GenericConstrainer::getRhs

  ( const Vector&  g,
    const Vector&  f ) const

{
  output_->getRhs ( g, f );
}


//-----------------------------------------------------------------------
//   getLhs
//-----------------------------------------------------------------------


void GenericConstrainer::getLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  output_->getLhs ( x, y );
}


//-----------------------------------------------------------------------
//   initLhs
//-----------------------------------------------------------------------


void GenericConstrainer::initLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  output_->initLhs ( x, y );
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* GenericConstrainer::getConstraints () const
{
  return output_->getConstraints ();
}


//-----------------------------------------------------------------------
//   getInputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* GenericConstrainer::getInputMatrix () const
{
  return output_->getInner ();
}


//-----------------------------------------------------------------------
//   getOutputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* GenericConstrainer::getOutputMatrix () const
{
  return output_.get ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Constrainer> GenericConstrainer::makeNew

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix )

{
  return newInstance<Self> ( name, cons, matrix );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GenericConstrainer::declare ()
{
  typedef ConstrainerFactory  Factory;

  Factory::declare ( TYPE_NAME,  & makeNew );
  Factory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( solver )
