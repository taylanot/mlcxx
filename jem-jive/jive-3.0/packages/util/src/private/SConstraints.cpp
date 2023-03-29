
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
#include <jem/base/limits.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/PrintWriter.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Reordering.h>
#include "SConstraints.h"


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class SConstraints
//=======================================================================

/*
  Implementation details.

  The field SlaveNode::idof stores both the rvalue mask and the index
  of a slave DOF. The rvalue mask is stored in the least significant
  bit, and the index and the remaining bit. This means that the
  expression

    (bool)  (SlaveNode::idof & JEM_UIDX_C(0x1))

  yields the rvalue mask, and the expression

    (idx_t) (SlaveNode::idof >> 1_uidx)

  yields the slave DOF index.

  The integer array dofConMap_ indicates for each DOF whether it is a
  slave, a master, or a regular DOF. Let idof denote the index of a
  DOF. If dofConMap_[idof] is larger than zero, then idof is a slave
  DOF and (dofConMap_[idof] - 1) is the index in the array slaves_ of
  the slave node. If dofConMap_[idof] is less than zero, then idof is
  a master DOF, and (-dofConMap_[idof] - 1) yields the index of the
  first slave DOF with which this master DOF is associated. If
  dofConMap_[idof] equals zero, then idof is a regular DOF.
*/

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SConstraints::SConstraints ( const Ref<DofSpace>& dofs ) :

  Super      ( dofs ),
  allocator_ ( sizeof(MasterNode) )

{
  masterCount_ = 0;

  newDofCount_ ();
}


SConstraints::SConstraints

  ( const Ref<DofSpace>&  dofs,
    const SparseMatrix&   cmat,
    const BoolVector&     rmask ) :

    Super      ( dofs ),
    allocator_ ( sizeof(MasterNode) )

{
  JEM_ASSERT2 ( cmat.size(0) == rmask.size() &&
                cmat.size(1) == dofCount_,
                "constraint matrix has an invalid shape" );

  IdxVector    offsets  = cmat.getRowOffsets    ();
  IdxVector    indices  = cmat.getColumnIndices ();
  Vector       values   = cmat.getValues        ();

  const idx_t  rowCount = cmat.size (0);


  masterCount_ = 0;

  newDofCount_    ();
  slaves_.reserve ( rowCount );

  try
  {
    for ( idx_t irow = 0; irow < rowCount; irow++ )
    {
      idx_t  i = offsets[irow];
      idx_t  n = offsets[irow + 1];

      idx_t  islave = newSlave_ ( indices[i], values[i], rmask[irow] );

      i++;

      if ( i < n )
      {
        newMaster_ ( slaves_[islave], indices[i], values[i] );
        i++;
      }

      if ( i < n )
      {
        throw jem::IllegalArgumentException (
          JEM_FUNC,
          "slave DOF has more than one master DOF"
        );
      }
    }
  }
  catch ( ... )
  {
    freeAll_ ();
    throw;
  }
}


SConstraints::SConstraints ( const Self& rhs ) :

  Super      ( rhs ),
  allocator_ ( sizeof(MasterNode) )

{
  const idx_t  slaveCount = rhs.slaves_.size ();

  masterCount_ = 0;

  newDofCount_    ();
  slaves_.reserve ( slaveCount );

  try
  {
    for ( idx_t i = 0; i < slaveCount; i++ )
    {
      newSlave_ ( rhs.slaves_[i] );
    }
  }
  catch ( ... )
  {
    freeAll_ ();
    throw;
  }
}


SConstraints::~SConstraints ()
{
  freeAll_ ();
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<ConstraintsImp> SConstraints::clone () const
{
  return jem::newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SConstraints::clear ()
{
  freeAll_ ();

  events .set   ( NEW_STRUCT );
  slaves_.clear ();

  dofConMap_   = 0;
  masterCount_ = 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void SConstraints::reserve ( idx_t slaveCount )
{
  slaves_.reserve ( slaveCount );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void SConstraints::trimToSize ()
{
  slaves_.trimToSize ();
}


//-----------------------------------------------------------------------
//   slaveDofCount
//-----------------------------------------------------------------------


idx_t SConstraints::slaveDofCount () const
{
  return slaves_.size ();
}


//-----------------------------------------------------------------------
//   masterDofCount
//-----------------------------------------------------------------------


idx_t SConstraints::masterDofCount () const
{
  return masterCount_;
}


idx_t SConstraints::masterDofCount ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  islave = dofConMap_[idof] - 1;

  if ( islave < 0 || slaves_[islave].master == 0 )
  {
    return 0;
  }
  else
  {
    return 1;
  }
}


//-----------------------------------------------------------------------
//   isSlaveDof
//-----------------------------------------------------------------------


bool SConstraints::isSlaveDof ( idx_t idof ) const
{
  if ( checkDofIndex( idof ) )
  {
    return (dofConMap_[idof] > 0);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   isMasterDof
//-----------------------------------------------------------------------


bool SConstraints::isMasterDof ( idx_t jdof ) const
{
  if ( checkDofIndex( jdof ) )
  {
    return (dofConMap_[jdof] < 0);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   hasRvalue
//-----------------------------------------------------------------------


bool SConstraints::hasRvalue ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return false;
  }

  const idx_t  islave = dofConMap_[idof] - 1;

  if ( islave < 0 )
  {
    return false;
  }
  else
  {
    return (bool) (slaves_[islave].idof & JEM_UIDX_C(0x1));
  }
}


//-----------------------------------------------------------------------
//   getSlaveDofs
//-----------------------------------------------------------------------


idx_t SConstraints::getSlaveDofs ( const IdxVector& idofs ) const
{
  const idx_t  n = jem::min ( slaves_.size(), idofs.size() );

  for ( idx_t i = 0; i < n; i++ )
  {
    idofs[i] = (idx_t) (slaves_[i].idof >> 1_uidx);
  }

  return n;
}


//-----------------------------------------------------------------------
//   getMasterDofs
//-----------------------------------------------------------------------


idx_t SConstraints::getMasterDofs

  ( const IdxVector&  jdofs,
    idx_t             idof ) const

{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  islave = dofConMap_[idof] - 1;

  if ( islave < 0 )
  {
    return 0;
  }

  const SlaveNode&  s = slaves_[islave];

  if ( s.master && jdofs.size() > 0 )
  {
    jdofs[0] = s.master->jdof;

    return 1;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   addConstraint (point)
//-----------------------------------------------------------------------


void SConstraints::addConstraint

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  checkDofIndex ( idof );

  idx_t  islave = dofConMap_[idof] - 1;
  idx_t  next;


  if ( islave >= 0 )
  {
    freeSlave_ ( islave );
  }

  next = -dofConMap_[idof] - 1;

  if ( next >= 0 )
  {
    uidx_t  umask = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);

    do
    {
      SlaveNode&  s = slaves_[dofConMap_[next] - 1];

      s.rvalue += s.master->coeff * rval;
      s.idof   |= umask;
      next      = s.master->next;

      freeMaster_ ( s );
    }
    while ( next >= 0 );

    dofConMap_[idof] = 0;
  }

  newSlave_  ( idof, rval, rmask );
  events.set ( NEW_STRUCT );
}


//-----------------------------------------------------------------------
//   addConstraint (simple)
//-----------------------------------------------------------------------


void SConstraints::addConstraint

  ( idx_t     idof,
    double  rval,
    bool    rmask,
    idx_t     jdof,
    double  coeff )

{
  checkDofIndex ( idof );
  checkDofIndex ( jdof );

  uidx_t  umask  = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);
  idx_t   islave = dofConMap_[idof] - 1;
  idx_t   next;


  if ( islave >= 0 )
  {
    freeSlave_ ( islave );
  }

  islave = dofConMap_[jdof] - 1;

  if ( islave >= 0 )
  {
    const SlaveNode&  s = slaves_[islave];

    rval  +=  s.rvalue * coeff;
    umask |= (s.idof   & JEM_UIDX_C(0x1));

    if ( s.master )
    {
      jdof   = s.master->jdof;
      coeff *= s.master->coeff;
    }
    else
    {
      jdof   = -1;
    }
  }

  if ( idof == jdof )
  {
    double  scale = 1.0 - coeff;

    if ( jem::isTiny( scale ) )
    {
      resolveError ( idof );
    }

    rval /=  scale;
    jdof  = -1;
  }

  next = -dofConMap_[idof] - 1;

  if ( next >= 0 )
  {
    MasterNode*  master = 0;

    do
    {
      SlaveNode&  s = slaves_[dofConMap_[next] - 1];

      master    = s.master;
      s.rvalue += master->coeff * rval;
      s.idof   |= umask;
      next      = master->next;

      if ( jdof < 0 )
      {
        freeMaster_ ( s );

        master = 0;
      }
      else
      {
        master->jdof   = jdof;
        master->coeff *= coeff;
      }
    }
    while ( next >= 0 );

    if ( jdof >= 0 )
    {
      JEM_ASSERT ( dofConMap_[jdof] <= 0 );

      master->next     = -dofConMap_[jdof] - 1;
      dofConMap_[jdof] =  dofConMap_[idof];
    }

    dofConMap_[idof] = 0;
  }

  islave = newSlave_ ( idof, rval, (umask != 0) );

  if ( jdof >= 0 )
  {
    newMaster_ ( slaves_[islave], jdof, coeff );
  }

  events.set ( NEW_STRUCT );
}


//-----------------------------------------------------------------------
//   eraseConstraint
//-----------------------------------------------------------------------


bool SConstraints::eraseConstraint ( idx_t idof )
{
  checkDofIndex ( idof );

  const idx_t  islave = dofConMap_[idof] - 1;

  if ( islave < 0 )
  {
    return false;
  }
  else
  {
    freeSlave_ ( islave );
    events.set ( NEW_STRUCT );

    return true;
  }
}


//-----------------------------------------------------------------------
//   eraseConstraints
//-----------------------------------------------------------------------


idx_t SConstraints::eraseConstraints ( const IdxVector& idofs )
{
  checkDofIndices ( idofs );

  const idx_t  n = idofs.size ();

  idx_t        islave;
  idx_t        i, k;


  k = 0;

  for ( i = 0; i < n; i++ )
  {
    islave = dofConMap_[idofs[i]] - 1;

    if ( islave >= 0 )
    {
      freeSlave_ ( islave );
      k++;
    }
  }

  if ( k > 0 )
  {
    events.set ( NEW_STRUCT );
  }

  return k;
}


//-----------------------------------------------------------------------
//   getConstraint
//-----------------------------------------------------------------------


idx_t SConstraints::getConstraint

  ( double&           rval,
    const IdxVector&  jdofs,
    const Vector&     coeffs,
    idx_t             idof ) const

{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  islave = dofConMap_[idof] - 1;

  if ( islave < 0 )
  {
    return 0;
  }

  const SlaveNode&  s = slaves_[islave];

  rval = s.rvalue;

  if ( s.master && jdofs.size() > 0 )
  {
    jdofs [0] = s.master->jdof;
    coeffs[0] = s.master->coeff;

    return 1;
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   setRvalue
//-----------------------------------------------------------------------


void SConstraints::setRvalue

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  checkDofIndex ( idof );

  const idx_t   islave = dofConMap_[idof] - 1;
  const uidx_t  umask  = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);


  if ( islave >= 0 )
  {
    SlaveNode&  s = slaves_[islave];

    s.rvalue = rval;

    if ( (s.idof & JEM_UIDX_C(0x1)) == umask )
    {
      if ( rmask )
      {
        events.set ( NEW_RVALS );
      }
    }
    else
    {
      events.set ( NEW_STRUCT );

      s.idof = (s.idof & ~JEM_UIDX_C(0x1)) | umask;
    }
  }
}


//-----------------------------------------------------------------------
//   setRvalues
//-----------------------------------------------------------------------


void SConstraints::setRvalues

  ( const IdxVector&  idofs,
    const Vector&     rvals )

{
  checkDofIndices ( idofs );

  const idx_t*  map = dofConMap_.addr ();
  const idx_t   n   = idofs     .size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  islave = map[idofs[i]] - 1;

    if ( islave >= 0 )
    {
      SlaveNode&  s = slaves_[islave];

      s.rvalue = rvals[i];

      if ( (s.idof & JEM_UIDX_C(0x1)) )
      {
        events.set ( NEW_RVALS );
      }
      else
      {
        events.set ( NEW_STRUCT );

        s.idof |= JEM_UIDX_C(0x1);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   unsetRvalues
//-----------------------------------------------------------------------


void SConstraints::unsetRvalues ( const IdxVector& idofs )
{
  checkDofIndices ( idofs );

  const idx_t*  map = dofConMap_.addr ();
  const idx_t   n   = idofs     .size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  islave = map[idofs[i]] - 1;

    if ( islave >= 0 )
    {
      SlaveNode&  s = slaves_[islave];

      if ( (s.idof & JEM_UIDX_C(0x1)) )
      {
        s.rvalue =  0.0;
        s.idof  &= ~JEM_UIDX_C(0x1);

        events.set ( NEW_STRUCT );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalue
//-----------------------------------------------------------------------


double SConstraints::getRvalue ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return 0.0;
  }

  const idx_t  islave = dofConMap_[idof] - 1;

  if ( islave < 0 )
  {
    return 0.0;
  }
  else
  {
    return slaves_[islave].rvalue;
  }
}


//-----------------------------------------------------------------------
//   getRvalues (with array arguments)
//-----------------------------------------------------------------------


void SConstraints::getRvalues

  ( const Vector&     rvals,
    const IdxVector&  idofs ) const

{
  const idx_t*  map = dofConMap_.addr ();
  const idx_t   n   = idofs     .size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  idof = idofs[i];

    if ( ! checkDofIndex( idof ) )
    {
      continue;
    }

    idx_t  islave = map[idof] - 1;

    if ( islave >= 0 )
    {
      rvals[i] = slaves_[islave].rvalue;
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalues (with a matrix argument)
//-----------------------------------------------------------------------


void SConstraints::getRvalues ( const SparseMatrix& cmat ) const
{
  IdxVector    offsets  = cmat.getRowOffsets    ();
  IdxVector    indices  = cmat.getColumnIndices ();
  Vector       values   = cmat.getValues        ();

  const idx_t  lastDof  = dofCount_ - 1;
  const idx_t  rowCount = cmat.size (0);


  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  i = offsets[irow];
    idx_t  n = offsets[irow + 1];

    if ( i >= n )
    {
      continue;
    }

    idx_t  idof = indices[i];

    if ( idof < 0 || idof > lastDof )
    {
      continue;
    }

    idx_t  islave = dofConMap_[idof] - 1;

    if ( islave >= 0 )
    {
      values[i] = slaves_[islave].rvalue;
    }
  }
}


//-----------------------------------------------------------------------
//   setSlaveDofs
//-----------------------------------------------------------------------


void SConstraints::setSlaveDofs

  ( const Vector&  vec,
    double         rscale ) const

{
  const SlaveNode*  slaves = slaves_.addr ();
  const idx_t       n      = slaves_.size ();


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      v[idof]     = slaves[i].rvalue * rscale;
    }
  }
  else
  {

JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t)   (slaves[i].idof >> 1_uidx);

      vec[idof]   = slaves[i].rvalue * rscale;
    }
  }
}


//-----------------------------------------------------------------------
//   evalSlaveDofs
//-----------------------------------------------------------------------


void SConstraints::evalSlaveDofs

  ( const Vector&  vec,
    double         rscale ) const

{
  const MasterNode*  master = 0;
  const SlaveNode*   slaves = slaves_.addr ();
  const idx_t        n      = slaves_.size ();


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      v[idof]     = slaves[i].rvalue * rscale;
      master      = slaves[i].master;

      if ( master )
      {
        v[idof] += master->coeff * v[master->jdof];
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      vec[idof]   = slaves[i].rvalue * rscale;
      master      = slaves[i].master;

      if ( master )
      {
        vec[idof] += master->coeff * vec[master->jdof];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   evalMasterDofs
//-----------------------------------------------------------------------


void SConstraints::evalMasterDofs ( const Vector& vec ) const
{
  const MasterNode*  master = 0;
  const SlaveNode*   slaves = slaves_.addr ();
  const idx_t        n      = slaves_.size ();


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      master      = slaves[i].master;

      if ( master )
      {
        v[master->jdof] += master->coeff * v[idof];
      }

      v[idof] = 0.0;
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      master      = slaves[i].master;

      if ( master )
      {
        vec[master->jdof] += master->coeff * vec[idof];
      }

      vec[idof] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalueMask
//-----------------------------------------------------------------------


BoolVector SConstraints::getRvalueMask () const
{
  const SlaveNode*  slaves = slaves_.addr ();
  const idx_t       n      = slaves_.size ();

  BoolVector        rmask ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    rmask[i] = (bool) (slaves[i].idof & JEM_UIDX_C(0x1));
  }

  return rmask;
}


void SConstraints::getRvalueMask

  ( const BoolVector&  rmask,
    const IdxVector&   idofs ) const

{
  const idx_t*  map = dofConMap_.addr ();
  const idx_t   n   = idofs     .size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  idof = idofs[i];

    if ( ! checkDofIndex( idof ) )
    {
      continue;
    }

    idx_t  islave = map[idof] - 1;

    if ( islave >= 0 )
    {
      rmask[i] = (bool) (slaves_[islave].idof & JEM_UIDX_C(0x1));
    }
  }
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseMatrix SConstraints::toMatrix () const
{
  const idx_t  dofCount   = dofs_->dofCount ();
  const idx_t  slaveCount = slaves_.size    ();

  IdxVector    offsets ( slaveCount + 1 );
  IdxVector    indices ( slaveCount + masterCount_ );
  Vector       values  ( slaveCount + masterCount_ );

  idx_t        j = 0;


  for ( idx_t islave = 0; islave < slaveCount; islave++ )
  {
    const SlaveNode&  s = slaves_[islave];

    offsets[islave] = j;

    indices[j] = (idx_t) (s.idof >> 1_uidx);
    values [j] = s.rvalue;

    j++;

    if ( s.master )
    {
      indices[j] = s.master->jdof;
      values [j] = s.master->coeff;
      j++;
    }
  }

  offsets[slaveCount] = j;

  return SparseMatrix ( jem::shape( slaveCount, dofCount ),
                        offsets, indices, values );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void SConstraints::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;

  const DofSpace&  dofs      = * dofs_;
  const ItemSet&   items     = * dofs .getItems   ();
  const ItemMap&   imap      = * items.getItemMap ();

  StringVector     typeNames = dofs.getTypeNames  ();

  idx_t            iitem;
  idx_t            itype;


  for ( idx_t islave = 0; islave < slaves_.size(); islave++ )
  {
    const SlaveNode&  s = slaves_[islave];

    idx_t  idof = (idx_t) (s.idof >> 1_uidx);

    dofs.decodeDofIndex ( iitem, itype, idof );

    idx_t  itemID = imap.getItemID ( iitem );

    if ( islave > 0 )
    {
      print ( out, endl );
    }

    print ( out, typeNames[itype], '[', itemID, "] = " );

    if ( ! s.master )
    {
      print ( out, s.rvalue );
    }
    else
    {
      double  coeff = s.master->coeff;

      idof = s.master->jdof;

      dofs.decodeDofIndex ( iitem, itype, idof );

      itemID = imap.getItemID ( iitem );

      if ( (s.idof & JEM_UIDX_C(0x1)) )
      {
        print ( out, s.rvalue );

        if ( coeff < 0.0 )
        {
          coeff *= -1.0;

          print ( out, " - " );
        }
        else
        {
          print ( out, " + " );
        }
      }

      print ( out, coeff, " * ",
              typeNames[itype], '[', itemID, ']' );
    }

    print ( out, ';' );
  }
}


//-----------------------------------------------------------------------
//   newDofCount
//-----------------------------------------------------------------------


void SConstraints::newDofCount ()
{
  const idx_t  oldCount = dofConMap_.size ();

  if ( dofCount_ == oldCount )
  {
    return;
  }

  if ( dofCount_ < oldCount )
  {
    const idx_t*  map = dofConMap_.addr ();

    idx_t         islave;
    idx_t         idof;
    idx_t         next;


    // Delete all invalid master nodes.

    for ( idof = dofCount_; idof < oldCount; idof++ )
    {
      next = -map[idof] - 1;

      while ( next >= 0 )
      {
        SlaveNode&  s = slaves_[map[next] - 1];

        next = s.master->next;

        freeMaster_ ( s );
      }
    }

    // Delete all invalid slave nodes.

    for ( idof = dofCount_; idof < oldCount; idof++ )
    {
      islave = map[idof] - 1;

      if ( islave >= 0 )
      {
        freeSlave_ ( islave );
      }
    }
  }

  newDofCount_ ();
}


//-----------------------------------------------------------------------
//   newDofOrder
//-----------------------------------------------------------------------


void SConstraints::newDofOrder ( const Reordering& reord )
{
  const idx_t*  JEM_RESTRICT  fmap = reord.fmapPtr ();

  idx_t         islave;
  idx_t         jslave;
  idx_t         idof;
  idx_t         jdof;


  dofConMap_.resize ( dofCount_ );

  dofConMap_ = 0;
  islave     = 0;

  while ( islave < slaves_.size() )
  {
    SlaveNode&  s = slaves_[islave];

    idof = (idx_t) (s.idof >> 1_uidx);
    idof = fmap[idof];

    if ( idof < 0 )
    {
      if ( s.master )
      {
        freeMaster_ ( s );
      }

      jslave = slaves_.size() - 1;

      if ( islave != jslave )
      {
        s = slaves_[jslave];
      }

      slaves_.popBack ();

      continue;
    }

    s.idof = ((uidx_t) idof << 1_uidx) |
              (s.idof & JEM_UIDX_C(0x1));

    if ( s.master )
    {
      jdof = fmap[s.master->jdof];

      if ( jdof < 0 )
      {
        freeMaster_ ( s );
      }
      else
      {
        s.master->jdof   =  jdof;
        s.master->next   = -dofConMap_[jdof] - 1;
        dofConMap_[jdof] = -idof - 1;
      }
    }

    JEM_ASSERT ( dofConMap_[idof] == 0 );

    islave++;

    dofConMap_[idof] = islave;
  }

  newDofCount_ ();
}


//-----------------------------------------------------------------------
//   freeAll_
//-----------------------------------------------------------------------


void SConstraints::freeAll_ ()
{
  const idx_t  n      = slaves_.size ();
  SlaveNode*   slaves = slaves_.addr ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( slaves[i].master )
    {
      allocator_.dealloc ( slaves[i].master );
    }
  }
}


//-----------------------------------------------------------------------
//   newSlave_
//-----------------------------------------------------------------------


idx_t SConstraints::newSlave_

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  JEM_ASSERT2 ( dofConMap_[idof] == 0,
                "DOF already has a constraint" );

  const idx_t   islave = slaves_.size ();
  const uidx_t  umask  = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);

  SlaveNode     slave;

  slave.idof   = ((uidx_t) idof << 1_uidx) | umask;
  slave.rvalue = rval;
  slave.master = 0;

  slaves_.pushBack ( std::move( slave ) );

  dofConMap_[idof] = islave + 1;

  return islave;
}


idx_t SConstraints::newSlave_ ( const SlaveNode& orig )
{
  const idx_t  idof   = (idx_t) (orig.idof >> 1_uidx);
  const idx_t  islave = slaves_.size ();

  SlaveNode    copy;

  copy.idof   = orig.idof;
  copy.rvalue = orig.rvalue;
  copy.master = 0;

  slaves_.pushBack ( std::move( copy ) );

  dofConMap_[idof] = islave + 1;

  if ( orig.master )
  {
    MasterNode*  master = orig.master;

    newMaster_ ( slaves_[islave], master->jdof, master->coeff );
  }

  return islave;
}


//-----------------------------------------------------------------------
//   freeSlave_
//-----------------------------------------------------------------------


void SConstraints::freeSlave_ ( idx_t islave )
{
  idx_t*      map   = dofConMap_.addr ();
  SlaveNode&  slave = slaves_[islave];

  idx_t       jslave;
  idx_t       idof;
  idx_t       jdof;
  idx_t       next;


  idof = (idx_t) (slave.idof >> 1_uidx);

  if ( slave.master )
  {
    jdof = slave.master->jdof;
    next = -map[jdof] - 1;

    if ( next == idof )
    {
      map[jdof] = -slave.master->next - 1;
    }
    else
    {
      jslave = map[next] - 1;
      next   = slaves_[jslave].master->next;

      while ( next != idof )
      {
        jslave = map[next] - 1;
        next   = slaves_[jslave].master->next;
      }

      slaves_[jslave].master->next = slave.master->next;
    }

    freeMaster_ ( slave );
  }

  map[idof] = 0;
  jslave    = slaves_.size() - 1;

  if ( islave != jslave )
  {
    slave     = slaves_[jslave];
    idof      = (idx_t) (slave.idof >> 1_uidx);
    map[idof] = islave + 1;
  }

  slaves_.popBack ();
}


//-----------------------------------------------------------------------
//   newMaster_
//-----------------------------------------------------------------------


inline void SConstraints::newMaster_

  ( SlaveNode&  slave,
    idx_t       jdof,
    double      coeff )

{
  JEM_ASSERT2 ( ! slave.master && dofConMap_[jdof] <= 0,
                "master DOF already defined" );

  MasterNode*  master = (MasterNode*) allocator_.alloc ();
  const idx_t  idof   = (idx_t) (slave.idof >> 1_uidx);


  master->jdof     =  jdof;
  master->coeff    =  coeff;
  master->next     = -dofConMap_[jdof] - 1;
  dofConMap_[jdof] = -idof - 1;
  slave.master     =  master;

  masterCount_++;
}


//-----------------------------------------------------------------------
//   freeMaster_
//-----------------------------------------------------------------------


inline void SConstraints::freeMaster_ ( SlaveNode& slave )
{
  JEM_ASSERT2 ( slave.master && masterCount_ > 0,
                "no such master DOF" );

  allocator_.dealloc ( slave.master );

  slave.master = 0;
  masterCount_--;
}


//-----------------------------------------------------------------------
//   newDofCount_
//-----------------------------------------------------------------------


void SConstraints::newDofCount_ ()
{
  const idx_t  oldCount = dofConMap_.size ();

  if      ( dofCount_ > oldCount )
  {
    dofConMap_.pushBack ( 0, dofCount_ - oldCount );
  }
  else if ( dofCount_ < oldCount )
  {
    dofConMap_.popBack  ( oldCount - dofCount_ );
  }

  events.set ( NEW_STRUCT );
}


JIVE_END_PACKAGE( util )
