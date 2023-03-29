
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
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/log.h>
#include <jive/util/Globdat.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include "private/CConstraints.h"
#include "private/SConstraints.h"
#include "private/XConstraints.h"


JEM_DEFINE_SERIABLE_CLASS( jive::util::Constraints );


JIVE_BEGIN_PACKAGE( util )


using jem::staticCast;
using jem::newInstance;


//=======================================================================
//   class Constraints
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Constraints::KIND_NAMES[] = { "COMPRESSED",
                                           "SIMPLE",
                                           "COMPLEX" };


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Constraints::Constraints ()
{
  ccons_ = 0;
  scons_ = 0;
  xcons_ = 0;
}


Constraints::Constraints

  ( const Ref<DofSpace>&  dofs,
    Kind                  kind )

{
  JEM_PRECHECK2 ( dofs, "NULL DofSpace" );

  ccons_ = 0;
  scons_ = 0;
  xcons_ = 0;

  switch ( kind )
  {
  case COMPRESSED:

    cons_  = newInstance<CConstraints> ( dofs );
    ccons_ = staticCast<CConstraints*> ( cons_ );

    break;

  case SIMPLE:

    cons_  = newInstance<SConstraints> ( dofs );
    scons_ = staticCast<SConstraints*> ( cons_ );

    break;

  default:

    cons_  = newInstance<XConstraints> ( dofs );
    xcons_ = staticCast<XConstraints*> ( cons_ );
  }

  connect_ ();
}


Constraints::Constraints ( const Self& rhs )
{
  cons_ = rhs.cons_->clone ();

  if      ( rhs.ccons_ )
  {
    ccons_ = staticCast<CConstraints*> ( cons_ );
  }
  else if ( rhs.scons_ )
  {
    scons_ = staticCast<SConstraints*> ( cons_ );
  }
  else if ( rhs.xcons_ )
  {
    xcons_ = staticCast<XConstraints*> ( cons_ );
  }

  cons_->events.reset ();

  connect_ ();
}


Constraints::~Constraints ()
{
  ccons_ = 0;
  scons_ = 0;
  xcons_ = 0;
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Constraints::toString () const
{
  if ( cons_ )
  {
    const String isetName = cons_->dofs_->getItems()->getName ();

    if ( isetName.size() > 0 )
    {
      return String::format ( "%s(%s)", CLASS_NAME, isetName );
    }
  }

  return Super::toString ();
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void Constraints::resetEvents ()
{
  cons_->events.reset       ();
  cons_->dofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void Constraints::readFrom ( ObjectInput& in )
{
  Ref<DofSpace>  dofs;
  SparseMatrix   cmat;
  BoolVector     rmask;


  decode ( in, dofs, cmat, rmask );

  cons_  = nullptr;
  cons_  = newInstance<CConstraints> ( dofs, cmat, rmask );
  ccons_ = staticCast<CConstraints*> ( cons_ );
  scons_ = 0;
  xcons_ = 0;

  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void Constraints::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           cons_->dofs_,
           cons_->toMatrix (),
           cons_->getRvalueMask () );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Constraints::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


void Constraints::copy ( const Constraints& rhs )
{
  cons_  = nullptr;
  ccons_ = 0;
  scons_ = 0;
  xcons_ = 0;

  cons_  = rhs.cons_->clone ();

  if      ( rhs.ccons_ )
  {
    ccons_ = staticCast<CConstraints*> ( cons_ );
  }
  else if ( rhs.scons_ )
  {
    scons_ = staticCast<SConstraints*> ( cons_ );
  }
  else if ( rhs.xcons_ )
  {
    xcons_ = staticCast<XConstraints*> ( cons_ );
  }

  cons_->events.reset ();

  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void Constraints::clear ()
{
  cons_->clear        ();
  cons_->events.emit  ( this );
  cons_->events.reset ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void Constraints::reserve ( idx_t slaveCount )
{
  if ( slaveCount > 0 )
  {
    cons_->reserve ( slaveCount );
  }
}


//-----------------------------------------------------------------------
//   compress
//-----------------------------------------------------------------------


void Constraints::compress ()
{
  if ( ! ccons_ )
  {
    realloc_ ( COMPRESSED );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void Constraints::trimToSize ()
{
  cons_->trimToSize ();
}


//-----------------------------------------------------------------------
//   dofCount
//-----------------------------------------------------------------------


idx_t Constraints::dofCount () const
{
  return cons_->dofs_->dofCount ();
}


//-----------------------------------------------------------------------
//   getSlaveDofs
//-----------------------------------------------------------------------


IdxVector Constraints::getSlaveDofs () const
{
  IdxVector  idofs ( cons_->slaveDofCount() );

  cons_->getSlaveDofs ( idofs );

  return idofs;
}


//-----------------------------------------------------------------------
//   setConstraints
//-----------------------------------------------------------------------


void Constraints::setConstraints ( const SparseMatrix& conmat )
{
  BoolVector  rmask ( conmat.size(0) );

  rmask = true;

  setConstraints ( conmat, rmask );
}


void Constraints::setConstraints

  ( const SparseMatrix&  cmat,
    const BoolVector&    rmask )

{
  JEM_PRECHECK2 ( cmat.size(0) == rmask.size() &&
                  cmat.size(1) == dofCount(),
                  "constraints matrix has wrong shape" );

  Ref<DofSpace>  dofs = cons_->dofs_;

  cons_  = newInstance<CConstraints> ( dofs, cmat, rmask );
  ccons_ = staticCast<CConstraints*> ( cons_ );
  scons_ = 0;
  xcons_ = 0;

  cons_->events.emit ( ConstraintsImp::NEW_STRUCT, this );
}


//-----------------------------------------------------------------------
//   eraseConstraint
//-----------------------------------------------------------------------


bool Constraints::eraseConstraint ( idx_t idof )
{
  bool  result;

 again:

  if      ( scons_ )
  {
    result = scons_->eraseConstraint ( idof );
  }
  else if ( xcons_ )
  {
    result = xcons_->eraseConstraint ( idof );
  }
  else
  {
    result = false;

    uncompress_ ();

    goto again;
  }

  cons_->events.emit ( this );

  return result;
}


//-----------------------------------------------------------------------
//   eraseConstraints
//-----------------------------------------------------------------------


idx_t Constraints::eraseConstraints ( const IdxVector& idofs )
{
  idx_t  n;

 again:

  if      ( scons_ )
  {
    n = scons_->eraseConstraints ( idofs );
  }
  else if ( xcons_ )
  {
    n = xcons_->eraseConstraints ( idofs );
  }
  else
  {
    n = 0;

    uncompress_ ();

    goto again;
  }

  cons_->events.emit ( this );

  return n;
}


//-----------------------------------------------------------------------
//   getConstraint
//-----------------------------------------------------------------------


idx_t Constraints::getConstraint

  ( double&           rval,
    const IdxVector&  jdofs,
    const Vector&     coeffs,
    idx_t             idof ) const

{
  JEM_PRECHECK2 ( jdofs.size() == coeffs.size(),
                  "Array size mismatch" );

  return cons_->getConstraint ( rval, jdofs, coeffs, idof );
}


//-----------------------------------------------------------------------
//   setRvalues
//-----------------------------------------------------------------------


void Constraints::setRvalues

  ( const IdxVector&  idofs,
    const Vector&     rvals )

{
  JEM_PRECHECK2 ( idofs.size() == rvals.size(),
                  "Array size mismatch" );

  cons_->setRvalues  ( idofs, rvals );
  cons_->events.emit ( this );
}


//-----------------------------------------------------------------------
//   unsetRvalues
//-----------------------------------------------------------------------


void Constraints::unsetRvalues ( const IdxVector& idofs )
{
  cons_->unsetRvalues ( idofs );
  cons_->events.emit  ( this  );
}


//-----------------------------------------------------------------------
//   getRvalues (with array arguments)
//-----------------------------------------------------------------------


void Constraints::getRvalues

  ( const Vector&     rvals,
    const IdxVector&  idofs ) const

{
  JEM_PRECHECK2 ( idofs.size() == rvals.size(),
                  "Array size mismatch" );

  cons_->getRvalues ( rvals, idofs );
}


//-----------------------------------------------------------------------
//   getRvalueMask
//-----------------------------------------------------------------------


void Constraints::getRvalueMask

  ( const BoolVector&  rmask,
    const IdxVector&   idofs ) const

{
  JEM_PRECHECK2 ( rmask.size() == idofs.size(),
                  "Array size mismatch" );

  cons_->getRvalueMask ( rmask, idofs );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void Constraints::printTo ( PrintWriter& out ) const
{
  cons_->printTo ( out );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void Constraints::store ( const Properties& globdat ) const
{
  Ref<Self>  self = const_cast<Self*> ( this );

  Globdat::storeFor ( Globdat::CONSTRAINTS, self,
                      cons_->dofs_, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<Constraints> Constraints::find

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<Self>  cons;

  Globdat::findFor ( cons, Globdat::CONSTRAINTS, dofs, globdat );

  return cons;
}


//-----------------------------------------------------------------------
//   extract
//-----------------------------------------------------------------------


Ref<Constraints> Constraints::extract

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<Self>  cons;

  Globdat::extractFor ( cons, Globdat::CONSTRAINTS, dofs, globdat );

  return cons;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Constraints> Constraints::get

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Properties  subdat = Globdat::getDataFor ( dofs, globdat );

  Ref<Self>   cons;

  if ( ! subdat.find( cons, Globdat::CONSTRAINTS ) )
  {
    cons = newInstance<Self> ( dofs );

    subdat.set ( Globdat::CONSTRAINTS, cons );
  }

  return cons;
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void Constraints::emitEvents_ ( int events )
{
  if      ( events & ConstraintsImp::NEW_STRUCT )
  {
    newStructEvent.emit ( *this );
  }
  else if ( events & ConstraintsImp::NEW_RVALS )
  {
    newRvaluesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void Constraints::connect_ ()
{
  using jem::util::connect;

  DofSpace&  dofs = * cons_->dofs_;

  connect ( dofs.newSizeEvent,  this, & Self::dofsResized_   );
  connect ( dofs.newOrderEvent, this, & Self::dofsReordered_ );
}


//-----------------------------------------------------------------------
//   realloc_
//-----------------------------------------------------------------------


void Constraints::realloc_ ( Kind kind )
{
  Ref<DofSpace>  dofs   = cons_->dofs_;
  SparseMatrix   cmat   = cons_->toMatrix      ();
  BoolVector     rmask  = cons_->getRvalueMask ();
  EventMask      events = cons_->events;


  cons_  = nullptr;
  ccons_ = 0;
  scons_ = 0;
  xcons_ = 0;

  switch ( kind )
  {
  case COMPRESSED:

    cons_  = newInstance<CConstraints> ( dofs, cmat, rmask );
    ccons_ = staticCast<CConstraints*> ( cons_ );

    break;

  case SIMPLE:

    cons_  = newInstance<SConstraints> ( dofs, cmat, rmask );
    scons_ = staticCast<SConstraints*> ( cons_ );

    break;

  default:

    cons_  = newInstance<XConstraints> ( dofs, cmat, rmask );
    xcons_ = staticCast<XConstraints*> ( cons_ );
  }

  cons_->events = events;

  print ( JIVE_LOG_DEBUG, toString(),
          " : reallocating constraints, kind = ",
          KIND_NAMES[kind], '\n' );
}


//-----------------------------------------------------------------------
//   uncompress_
//-----------------------------------------------------------------------


void Constraints::uncompress_ ()
{
  if ( ccons_ )
  {
    if ( ccons_->maxMasterCount() > 1 )
    {
      realloc_ ( COMPLEX );
    }
    else
    {
      realloc_ ( SIMPLE );
    }
  }
}


void Constraints::uncompress_ ( Kind kind )
{
  if ( kind == SIMPLE && ccons_ )
  {
    if ( ccons_->maxMasterCount() > 1 )
    {
      realloc_ ( COMPLEX );
    }
    else
    {
      realloc_ ( SIMPLE );
    }

    return;
  }

  if ( kind == COMPLEX && ! xcons_ )
  {
    realloc_ ( COMPLEX );

    return;
  }
}


//-----------------------------------------------------------------------
//   addConstraint_ (point)
//-----------------------------------------------------------------------


void Constraints::addConstraint_

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
 again:

  if      ( scons_ )
  {
    scons_->addConstraint ( idof, rval, rmask );
  }
  else if ( xcons_ )
  {
    xcons_->addConstraint ( idof, rval, rmask );
  }
  else
  {
    uncompress_ ( SIMPLE );

    goto again;
  }

  cons_->events.emit ( this );
}


//-----------------------------------------------------------------------
//   addConstraint_ (simple)
//-----------------------------------------------------------------------


void Constraints::addConstraint_

  ( idx_t   idof,
    double  rval,
    bool    rmask,
    idx_t   jdof,
    double  coeff )

{
 again:

  if      ( scons_ )
  {
    scons_->addConstraint ( idof, rval, rmask, jdof, coeff );
  }
  else if ( xcons_ )
  {
    xcons_->addConstraint ( idof, rval, rmask, jdof, coeff );
  }
  else
  {
    uncompress_ ( SIMPLE );

    goto again;
  }

  cons_->events.emit ( this );
}


//-----------------------------------------------------------------------
//   addConstraint_ (linear)
//-----------------------------------------------------------------------


void Constraints::addConstraint_

  ( idx_t             idof,
    double            rval,
    bool              rmask,
    const IdxVector&  jdofs,
    const Vector&     coeffs )

{
  JEM_PRECHECK2 ( jdofs.size() == coeffs.size(),
                  "Array size mismatch" );

  const idx_t  n = jdofs.size ();

 again:

  if      ( xcons_ )
  {
    xcons_->addConstraint ( idof, rval, rmask, jdofs, coeffs );
  }
  else if ( scons_ && n < 2 )
  {
    if ( n == 0 )
    {
      scons_->addConstraint ( idof, rval, rmask );
    }
    else
    {
      scons_->addConstraint ( idof, rval, rmask,
                              jdofs[0], coeffs[0] );
    }
  }
  else
  {
    Kind  kind = (n < 2) ? SIMPLE : COMPLEX;

    uncompress_ ( kind );

    goto again;
  }

  cons_->events.emit ( this );
}


//-----------------------------------------------------------------------
//   dofsResized_
//-----------------------------------------------------------------------


void Constraints::dofsResized_ ( idx_t count )
{
  if ( count == 0 )
  {
    clear ();
  }

  cons_->dofCount_ = count;

  cons_->newDofCount ();
  cons_->events.emit ( this );
}


//-----------------------------------------------------------------------
//   dofsReordered_
//-----------------------------------------------------------------------


void Constraints::dofsReordered_ ( const Reordering& reord )
{
  cons_->dofCount_ = reord.newSize ();

  cons_->newDofOrder ( reord );
  cons_->events.emit ( this  );
}


JIVE_END_PACKAGE( util )
