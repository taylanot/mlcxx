
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


#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/Float.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/intrinsics.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/log.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Reordering.h>
#include <jive/util/Constraints.h>
#include <jive/util/SparseDofSpace.h>
#include <jive/algebra/ConDistiller.h>


JEM_DEFINE_SERIABLE_CLASS( jive::algebra::ConDistiller );


using jem::newInstance;
using jive::util::ItemSet;


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   private macros
//=======================================================================


#undef  JIVE_CHECK_VECTORS
#define JIVE_CHECK_VECTORS( f, r )                      \
                                                        \
  if ( f.size() != dofCount_ )                          \
    util::sizeError ( JEM_FUNC, "full vector",          \
                      f.size(), dofCount_ );            \
                                                        \
  if ( r.size() != rdofCount_ )                         \
    util::sizeError ( JEM_FUNC, "reduced vector",       \
                      r.size(), rdofCount_ )


//=======================================================================
//   class ConDistiller
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ConDistiller::NEW_STRUCT_  = 1 << 0;
const int  ConDistiller::NEW_RVALUES_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ConDistiller::ConDistiller ()
{
  dofCount_  = -1;
  rdofCount_ = -1;
  dirtyBits_ =  NEW_STRUCT_;
}


ConDistiller::ConDistiller

  ( Ref<Constraints>   cons,
    Ref<XDofSpace>     rdofs ) :

    cons_  (  cons ),
    rdofs_ ( rdofs )

{
  using jive::util::SparseDofSpace;

  JEM_PRECHECK2 ( cons, "NULL Constraints" );

  ItemSet*  items = cons->getDofSpace()->getItems ();

  if      ( ! rdofs_ )
  {
    rdofs_ = newInstance<SparseDofSpace> ( items );
  }
  else if ( items != rdofs->getItems() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "constraints and reduced dof space are associated with "
      "different item sets"
    );
  }

  rcons_     = newInstance<Constraints> ( rdofs_ );
  dirtyBits_ = 0;

  structChanged_ ();
  connect_       ();
}


ConDistiller::~ConDistiller ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ConDistiller::readFrom ( ObjectInput& in )
{
  using jem::checkedCast;

  cons_  = checkedCast<Constraints> ( in.decodeObject() );
  rcons_ = checkedCast<Constraints> ( in.decodeObject() );
  rdofs_ = checkedCast<XDofSpace>   ( in.decodeObject() );

  structChanged_ ();
  connect_       ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ConDistiller::writeTo ( ObjectOutput& out ) const
{
  using jem::checkedCast;

  out.encodeObject ( cons_  );
  out.encodeObject ( rcons_ );
  out.encodeObject ( rdofs_ );
}


//-----------------------------------------------------------------------
//   reduce
//-----------------------------------------------------------------------


void ConDistiller::reduce

  ( const Vector&  r,
    const Vector&  f ) const

{
  r = 0.0;

  rdaxpy ( r, 1.0, f );
}


//-----------------------------------------------------------------------
//   rdaxpy
//-----------------------------------------------------------------------


void ConDistiller::rdaxpy

  ( const Vector&  r,
    double         s,
    const Vector&  f ) const

{
  JEM_PRECHECK2 ( updated(), "ConDistiller has not been updated" );

  const idx_t*   offsets = conOffsets_.addr ();
  const idx_t*   indices = dofIndices_.addr ();
  const double*  coeffs  = conCoeffs_ .addr ();

  const idx_t    n = dofCount_;

  double         t;
  idx_t          i, j, k;


  JIVE_CHECK_VECTORS( f, r );

  for ( i = j = 0; i < n; i++ )
  {
    k = offsets[i + 1];
    t = s * f[i];

JEM_IVDEP

    for ( ; j < k; j++ )
    {
      r[indices[j]] += t * coeffs[j];
    }
  }
}


//-----------------------------------------------------------------------
//   filter
//-----------------------------------------------------------------------


void ConDistiller::filter

  ( const Vector&  r,
    const Vector&  f ) const

{
  JEM_PRECHECK2 ( updated(), "ConDistiller has not been updated" );

  const idx_t*  dofMap = dofMap_.addr ();

  const idx_t   n      = dofCount_;

  idx_t         i, j;


  JIVE_CHECK_VECTORS( f, r );

  for ( i = 0; i < n; i++ )
  {
    j = dofMap[i];

    if ( j >= 0 )
    {
      r[j] = f[i];
    }
  }
}


//-----------------------------------------------------------------------
//   expand
//-----------------------------------------------------------------------


void ConDistiller::expand

  ( const Vector&  f,
    const Vector&  r ) const

{
  JEM_PRECHECK2 ( updated(), "ConDistiller has not been updated" );

  const idx_t*   offsets = conOffsets_.addr ();
  const idx_t*   indices = dofIndices_.addr ();
  const double*  coeffs  = conCoeffs_ .addr ();

  const idx_t    n       = dofCount_;

  double         t;
  idx_t          i, j, k;


  JIVE_CHECK_VECTORS( f, r );

  for ( i = j = 0; i < n; i++ )
  {
    k = offsets[i + 1];
    t = 0.0;

    for ( ; j < k; j++ )
    {
      t += r[indices[j]] * coeffs[j];
    }

    f[i] = t;
  }
}


//-----------------------------------------------------------------------
//   getDofMap
//-----------------------------------------------------------------------


IdxVector ConDistiller::getDofMap () const
{
  return dofMap_;
}


//-----------------------------------------------------------------------
//   getDofSpace
//-----------------------------------------------------------------------


DofSpace* ConDistiller::getDofSpace () const
{
  return cons_->getDofSpace ();
}


//-----------------------------------------------------------------------
//   getReducedDofs
//-----------------------------------------------------------------------


DofSpace* ConDistiller::getReducedDofs () const
{
  return rdofs_.get ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* ConDistiller::getConstraints () const
{
  return cons_.get ();
}


//-----------------------------------------------------------------------
//   getReducedCons
//-----------------------------------------------------------------------


Constraints* ConDistiller::getReducedCons () const
{
  return rcons_.get ();
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void ConDistiller::connect_ ()
{
  using jem::util::connect;

  DofSpace*  dofs = cons_->getDofSpace ();

  connect ( dofs  ->newSizeEvent,    this, & Self::structChanged_  );
  connect ( dofs  ->newOrderEvent,   this, & Self::structChanged_  );
  connect ( cons_ ->newStructEvent,  this, & Self::structChanged_  );
  connect ( cons_ ->newRvaluesEvent, this, & Self::rvaluesChanged_ );
  connect ( rdofs_->newSizeEvent,    this, & Self::rdofsResized_   );
  connect ( rdofs_->newOrderEvent,   this, & Self::rdofsReordered_ );

  dofs  ->resetEvents ();
  cons_ ->resetEvents ();
  rdofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void ConDistiller::update_ ()
{
  DofSpace*  dofs = cons_->getDofSpace ();

  if      ( dirtyBits_ & NEW_STRUCT_ )
  {
    updateStruct_  ();
  }
  else if ( dirtyBits_ & NEW_RVALUES_ )
  {
    updateRvalues_ ();
  }

  dirtyBits_ = 0;

  dofs  ->resetEvents ();
  cons_ ->resetEvents ();
  rdofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   updateStruct_
//-----------------------------------------------------------------------


void ConDistiller::updateStruct_ ()
{
  using jem::where;

  DofSpace*    dofs     = cons_->getDofSpace   ();

  const idx_t  conCount = cons_->slaveDofCount ();

  BoolVector   rmask    ( conCount );
  IdxVector    idofs    ( conCount );
  IdxVector    jdofs;
  Vector       coeffs;

  double       chksum;
  double       rval;
  idx_t        iitem;
  idx_t        jtype;
  idx_t        idof;
  idx_t        jdof;
  idx_t        i, j, k, m, n;


  print ( JIVE_LOG_DEBUG, CLASS_NAME,
          " : updating reduced dof space ...\n" );

  cons_ ->compress ();
  rcons_->clear    ();
  rdofs_->clearAll ();
  rdofs_->addTypes ( dofs->getTypeNames() );

  cons_->getSlaveDofs  (        idofs );
  cons_->getRvalueMask ( rmask, idofs );

  dofCount_ = dofs->dofCount ();

  dofMap_.resize ( dofCount_ );

  dofMap_ = 0;

  for ( i = 0; i < conCount; i++ )
  {
    dofMap_[idofs[i]] = rmask[i] ? 0 : -1;
  }

  for ( idof = 0; idof < dofCount_; idof++ )
  {
    if ( dofMap_[idof] < 0 )
    {
      continue;
    }

    dofs->decodeDofIndex ( iitem, jtype, idof );

    dofMap_[idof] = rdofs_->addDof ( iitem, jtype );
  }

  n = rdofCount_ = rdofs_->dofCount ();
  m = 0;

  for ( i = 0; i < conCount; i++ )
  {
    k = cons_->masterDofCount ( idofs[i] );

    if ( k > m )
    {
      m = k;
    }

    if ( ! rmask[i] )
    {
      n += k;
    }
  }

  conOffsets_.resize ( dofCount_ + 1 );
  dofIndices_.resize ( n );
  conCoeffs_ .resize ( n );
  coeffs     .resize ( m );
  jdofs      .resize ( m );

  chksum = 0.0;

  for ( idof = k = 0; idof < dofCount_; idof++ )
  {
    conOffsets_[idof] = k;

    jdof = dofMap_[idof];

    if ( jdof >= 0 )
    {
      dofIndices_[k] = jdof;
      conCoeffs_ [k] = 1.0;
      k++;
    }
    else
    {
      n = cons_->getConstraint ( rval, jdofs, coeffs, idof );

      chksum += rval;

      for ( j = 0; j < n; j++, k++ )
      {
        jdof = dofMap_[jdofs[j]];

        JEM_ASSERT ( jdof >= 0 );

        dofIndices_[k] = jdof;
        conCoeffs_ [k] = coeffs[j];
        chksum        += coeffs[j];
      }
    }
  }

  conOffsets_[dofCount_] = k;

  // Transfer the constraints with non-zero rvalue

  for ( i = 0; i < conCount; i++ )
  {
    if ( ! rmask[i] )
    {
      continue;
    }

    idof = idofs[i];
    jdof = dofMap_[idof];

    k = cons_->getConstraint ( rval, jdofs, coeffs, idof );

    if ( k == 0 )
    {
      rcons_->addConstraint ( jdof, rval );
    }
    else
    {
      for ( j = 0; j < k; j++ )
      {
        jdofs[j] = dofMap_[jdofs[j]];
      }

      rcons_->addConstraint ( jdof, rval, jdofs, coeffs );
    }
  }

  if ( jem::Float::isNaN( chksum ) )
  {
    nanError_ ();
  }

  printInfo_ ();
}


//-----------------------------------------------------------------------
//   updateRvalues_
//-----------------------------------------------------------------------


void ConDistiller::updateRvalues_ ()
{
  const idx_t  conCount = cons_->slaveDofCount ();

  IdxVector    idofs    ( conCount );
  BoolVector   rmask    ( conCount );
  Vector       rvals;

  idx_t        i, k;


  cons_->getSlaveDofs  (        idofs );
  cons_->getRvalueMask ( rmask, idofs );

  // Remove all constrained dofs with a non-zero rvalue

  for ( i = k = 0; i < conCount; i++ )
  {
    if ( rmask[i] )
    {
      idofs[k++] = idofs[i];
    }
  }

  idofs.resize ( k );
  rvals.resize ( k );

  cons_->getRvalues ( rvals, idofs );

  for ( i = 0; i < k; i++ )
  {
    idofs[i] = dofMap_[idofs[i]];
  }

  if ( rvals.size() > 0 )
  {
    double  chksum = jem::sum ( rvals );

    if ( jem::Float::isNaN( chksum ) )
    {
      nanError_ ();
    }
  }

  rcons_->setRvalues ( idofs, rvals );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void ConDistiller::structChanged_ ()
{
  dofCount_   = -1;
  rdofCount_  = -1;
  dirtyBits_ |=  NEW_STRUCT_;

  if ( dofMap_.size() )
  {
    // Free memory

    dofMap_    .ref ( IdxVector() );
    conOffsets_.ref ( dofMap_     );
    dofIndices_.ref ( dofMap_     );
    conCoeffs_ .ref ( Vector()    );
  }
}


//-----------------------------------------------------------------------
//   rvaluesChanged_
//-----------------------------------------------------------------------


void ConDistiller::rvaluesChanged_ ()
{
  dirtyBits_ |= NEW_RVALUES_;
}


//-----------------------------------------------------------------------
//   rdofsResized_
//-----------------------------------------------------------------------


void ConDistiller::rdofsResized_ ( idx_t size )
{
  rdofCount_ = rdofs_->dofCount ();

  if ( rdofCount_ == 0 && ! (dirtyBits_ & NEW_STRUCT_) )
  {
    dofMap_     = -1;
    conOffsets_ =  0;

    if ( dofIndices_.size() )
    {
      dofIndices_.resize ( 0 );
      conCoeffs_ .resize ( 0 );
    }
  }

  if ( ! (dirtyBits_ & NEW_STRUCT_) )
  {
    rdofs_->resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   rdofsReordered_
//-----------------------------------------------------------------------


void ConDistiller::rdofsReordered_ ( const Reordering& reord )
{
  if ( dirtyBits_ == 0 )
  {
    const idx_t*  fmap     = reord.fmapPtr ();

    const idx_t   dofCount = dofMap_.size  ();

    idx_t*        dofMap   = dofMap_    .addr ();
    idx_t*        offsets  = conOffsets_.addr ();
    idx_t*        indices  = dofIndices_.addr ();
    double*       coeffs   = conCoeffs_ .addr ();

    idx_t         idof;
    idx_t         jdof;
    idx_t         j, k, n;


    for ( idof = j = k = 0; idof < dofCount; idof++ )
    {
      offsets[idof] = k;
      n = offsets[idof + 1];

      for ( ; j < n; j++ )
      {
        jdof = fmap[indices[j]];

        if ( jdof >= 0 )
        {
          indices[k] = jdof;
          coeffs [k] = coeffs[j];
          k++;
        }
      }

      jdof = dofMap[idof];

      if ( jdof >= 0 )
      {
        dofMap[idof] = fmap[jdof];
      }
    }

    offsets[dofCount] = k;

    dofIndices_.resize ( k );
    conCoeffs_ .resize ( k );
  }
}


//-----------------------------------------------------------------------
//   printInfo_
//-----------------------------------------------------------------------


void ConDistiller::printInfo_ ()
{
  jem::io::Writer&     out = JIVE_LOG_DEBUG;

  const idx_t  slaveCount  = cons_ ->slaveDofCount ();
  const idx_t  rslaveCount = rcons_->slaveDofCount ();

  print ( out, "\n  original dofs .......... ", dofCount_   );
  print ( out, "\n  reduced  dofs .......... ", rdofCount_  );
  print ( out, "\n  original constraints ... ", slaveCount  );
  print ( out, "\n  reduced  constraints ... ", rslaveCount );
  print ( out, "\n\n" );
}


//-----------------------------------------------------------------------
//   nanError_
//-----------------------------------------------------------------------


void ConDistiller::nanError_ ()
{
  throw jem::ArithmeticException (
    CLASS_NAME,
    "invalid constraints (NaN)"
  );
}


JIVE_END_PACKAGE( algebra )
