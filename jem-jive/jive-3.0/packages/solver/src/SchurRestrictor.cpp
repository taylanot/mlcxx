
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


#include <jem/base/Array.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/util/Event.h>
#include <jive/solver/SchurMatrix.h>
#include <jive/solver/LocalRestrictor.h>
#include <jive/solver/SchurRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::SchurRestrictor );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class SchurRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SchurRestrictor::SchurRestrictor

  ( Ref<Restrictor>   inner,
    Ref<SchurMatrix>  schur ) :

    inner_ ( inner ),
    schur_ ( schur )

{
  using jem::IllegalArgumentException;

  JEM_PRECHECK ( inner && schur );

  if ( ! inner->getExtension<LocalRestrictor>() )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      inner->getContext() + " is not a local restrictor"
    );
  }

  myName_ = inner->getName ();

  connect_ ();
}


SchurRestrictor::~SchurRestrictor ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SchurRestrictor::start ()
{
  inner_->start ();
  schur_->start ();
  Super ::start ();
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SchurRestrictor::finish ()
{
  inner_->finish ();
  schur_->finish ();
  Super ::finish ();
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void SchurRestrictor::update ()
{
  inner_->update ();
  schur_->update ();
  Super ::update ();
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SchurRestrictor::resetEvents ()
{
  inner_->resetEvents ();
  schur_->resetEvents ();
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SchurRestrictor::configure ( const Properties& props )
{
  inner_->configure ( props );
  schur_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SchurRestrictor::getConfig ( const Properties& conf ) const
{
  inner_->getConfig ( conf );
  schur_->getConfig ( conf );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SchurRestrictor::connect_ ()
{
  using jem::util::connect;

  inner_->resetEvents ();
  schur_->resetEvents ();

  connect ( inner_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( schur_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( inner_->newStructEvent, this, & Self::structChanged_ );
  connect ( schur_->newStructEvent, this, & Self::structChanged_ );
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SchurRestrictor::update_ ( int events )
{
  LocalRestrictor*  rt = inner_->getExtension<LocalRestrictor> ();

  SchurDofSpace*  schurDofs  = schur_->getSchurDofSpace ();
  IdxVector       borderDofs = schurDofs->getBorderDofs ();

  const idx_t     basisSize  = inner_->size   (0);
  const idx_t     totalSize  = inner_->size   (1);
  const idx_t     schurSize  = borderDofs.size ();

  Vector          v ( totalSize );


  // Copy the border DOFs from the basis spanned by the inner
  // restriction operator.

  basis_.resize ( schurSize, basisSize );

  for ( idx_t j = 0; j < basisSize; j++ )
  {
    rt->getBasis ( v, j );

    basis_[j] = v[borderDofs];
  }
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SchurRestrictor::valuesChanged_ ()
{
  setEvents_          ( NEW_VALUES_ | NEW_STRUCT_ );
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SchurRestrictor::structChanged_ ()
{
  setEvents_          ( NEW_VALUES_ | NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
