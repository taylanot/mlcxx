
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
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Reordering.h>
#include "XConstraints.h"


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class XConstraints
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

  The integer array slaveMap_ maps DOF indices to slave nodes. To be
  precise, if slaveMap_[idof] is larger than zero, then the DOF with
  index idof is a slave DOF and (slaveMap_[idof] - 1) is the index in
  the array slaves_ of the slave node. Otherwise, if slaveMap_[idof]
  is non-positive, then the DOF with index idof is not a slave DOF.

  The array slaveMap_ is also used to map master DOFs to entries in
  the scratch arrays in the method addConstraints. This is possible
  because a DOF is either a master or a slave, but not both.
*/

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


XConstraints::XConstraints ( const Ref<DofSpace>& dofs ) :

  Super      ( dofs ),
  allocator_ ( sizeof(MasterNode) )

{
  masterCount_ = 0;

  newDofCount_ ();
}


XConstraints::XConstraints

  ( const Ref<DofSpace>&  dofs,
    const SparseMatrix&   cmat,
    const BoolVector&     rmask ) :

    Super      ( dofs ),
    allocator_ ( sizeof(MasterNode) )

{
  JEM_ASSERT2 ( cmat.size(0) == rmask.size() &&
                cmat.size(1) == dofCount_,
                "constraints matrix has an invalid shape" );

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
      idx_t  i      = offsets[irow];
      idx_t  n      = offsets[irow + 1];

      idx_t  islave = newSlave_ ( indices[i], values[i], rmask[irow] );

      for ( i++; i < n; i++ )
      {
        newMaster_ ( islave, indices[i], values[i] );
      }
    }
  }
  catch ( ... )
  {
    freeAll_ ();
    throw;
  }
}


XConstraints::XConstraints ( const Self& rhs ) :

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


XConstraints::~XConstraints ()
{
  freeAll_ ();
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<ConstraintsImp> XConstraints::clone () const
{
  return jem::newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void XConstraints::clear ()
{
  Scratch_&  s = scratch_;

  freeAll_ ();

  slaves_ .clear ();
  s.jdofs .clear ();
  s.coeffs.clear ();
  s.accu  .clear ();

  events .set ( NEW_STRUCT );

  masterCount_ = 0;
  slaveMap_    = 0;
  masterMap_   = (MasterNode*) 0;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void XConstraints::reserve ( idx_t slaveCount )
{
  slaves_.reserve ( slaveCount );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void XConstraints::trimToSize ()
{
  Scratch_&  s = scratch_;

  slaves_   .trimToSize ();
  slaveMap_ .trimToSize ();
  masterMap_.trimToSize ();

  s.jdofs   .clear      ();
  s.jdofs   .trimToSize ();
  s.coeffs  .clear      ();
  s.coeffs  .trimToSize ();
  s.accu    .clear      ();
  s.accu    .trimToSize ();
}


//-----------------------------------------------------------------------
//   slaveDofCount
//-----------------------------------------------------------------------


idx_t XConstraints::slaveDofCount () const
{
  return slaves_.size ();
}


//-----------------------------------------------------------------------
//   masterDofCount
//-----------------------------------------------------------------------


idx_t XConstraints::masterDofCount () const
{
  return masterCount_;
}


idx_t XConstraints::masterDofCount ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  islave = slaveMap_[idof] - 1;
  idx_t        count  = 0;

  if ( islave >= 0 )
  {
    MasterNode*  master = slaves_[islave].masterList;

    while ( master )
    {
      master = master->next;
      count++;
    }
  }

  return count;
}


//-----------------------------------------------------------------------
//   isSlaveDof
//-----------------------------------------------------------------------


bool XConstraints::isSlaveDof ( idx_t idof ) const
{
  if ( checkDofIndex( idof ) )
  {
    return (slaveMap_[idof] > 0);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   isMasterDof
//-----------------------------------------------------------------------


bool XConstraints::isMasterDof ( idx_t jdof ) const
{
  if ( checkDofIndex( jdof ) )
  {
    return (masterMap_[jdof] != 0);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   hasRvalue
//-----------------------------------------------------------------------


bool XConstraints::hasRvalue ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return false;
  }

  const idx_t  islave = slaveMap_[idof] - 1;

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


idx_t XConstraints::getSlaveDofs ( const IdxVector& idofs ) const
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


idx_t XConstraints::getMasterDofs

  ( const IdxVector&  jdofs,
    idx_t             idof ) const

{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  n = jdofs.size ();

  MasterNode*  master;
  idx_t        islave;
  idx_t        j;


  islave = slaveMap_[idof] - 1;

  if ( islave < 0 )
  {
    return 0;
  }

  master = slaves_[islave].masterList;
  j      = 0;

  while ( master && j < n )
  {
    jdofs[j++] = master->jdof;
    master     = master->next;
  }

  return j;
}


//-----------------------------------------------------------------------
//   addConstraint (point)
//-----------------------------------------------------------------------


void XConstraints::addConstraint

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  checkDofIndex ( idof );

  MasterNode*  master;
  MasterNode*  up;


  if ( slaveMap_[idof] > 0 )
  {
    freeSlave_ ( slaveMap_[idof] - 1 );
  }

  master = masterMap_[idof];

  if ( master )
  {
    uidx_t  umask = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);

    do
    {
      SlaveNode&  s = slaves_[master->islave];

      up        = master->up;

      s.rvalue += master->coeff * rval;
      s.idof   |= umask;

      freeMaster_ ( master );

      master    = up;
    }
    while ( master );
  }

  newSlave_  ( idof, rval, rmask );
  events.set ( NEW_STRUCT );
}


//-----------------------------------------------------------------------
//   addConstraint (simple)
//-----------------------------------------------------------------------


void XConstraints::addConstraint

  ( idx_t   idof,
    double  rval,
    bool    rmask,
    idx_t   jdof,
    double  coeff )

{
  checkDofIndex ( idof );
  checkDofIndex ( jdof );


  if ( slaveMap_[idof] > 0 )
  {
    freeSlave_ ( slaveMap_[idof] - 1 );
  }

  // Check whether the constraint can be added directly without
  // substitutions (the common case).

  if ( masterMap_[idof] == 0 &&
       slaveMap_ [jdof] == 0 && idof != jdof )
  {
    idx_t  k = newSlave_ ( idof, rval, rmask );

    newMaster_ ( k, jdof, coeff );
    events.set ( NEW_STRUCT );

    return;
  }

  try
  {
    addConstraint_ ( idof, rval, rmask, &jdof, &coeff, 1 );
  }
  catch ( ... )
  {
    fixSlaveMap_ ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   addConstraint (linear)
//-----------------------------------------------------------------------


void XConstraints::addConstraint

  ( idx_t             idof,
    double            rval,
    bool              rmask,
    const IdxVector&  jdofs,
    const Vector&     coeffs )

{
  using jem::makeContiguous;

  checkDofIndex   ( idof  );
  checkDofIndices ( jdofs );

  const idx_t  jcount = jdofs    .size ();
  idx_t*       map    = slaveMap_.addr ();

  idx_t        jdof;
  idx_t        j, k;


  if ( map[idof] > 0 )
  {
    freeSlave_ ( map[idof] - 1 );
  }

  // Check whether the constraint can be added directly without
  // substitutions (the common case).

  if ( masterMap_[idof] )
  {
    goto resolve;
  }

  map[idof] = -1;

  for ( j = k = 0; j < jcount; j++ )
  {
    jdof = jdofs[j];

    if ( map[jdof] > 0 )
    {
      k++;
    }
    else
    {
      k        += -map[jdof];
      map[jdof] = -1;
    }
  }

  map[idof] = 0;

  for ( j = 0; j < jcount; j++ )
  {
    jdof = jdofs[j];

    if ( map[jdof] < 0 )
    {
      map[jdof] = 0;
    }
  }

  if ( k > 0 )
  {
    goto resolve;
  }

  k = newSlave_ ( idof, rval, rmask );

  for ( j = 0; j < jcount; j++ )
  {
    newMaster_ ( k, jdofs[j], coeffs[j] );
  }

  events.set ( NEW_STRUCT );

  return;


 resolve:

  try
  {
    if ( jdofs.stride() + coeffs.stride() == 2L )
    {
      addConstraint_ ( idof, rval, rmask,
                       jdofs.addr(), coeffs.addr(), jcount );
    }
    else
    {
      Vector     cc = makeContiguous ( coeffs );
      IdxVector  jj = makeContiguous ( jdofs  );

      addConstraint_ ( idof, rval, rmask,
                       jj.addr(), cc.addr(), jcount );
    }
  }
  catch ( ... )
  {
    fixSlaveMap_ ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   eraseConstraint
//-----------------------------------------------------------------------


bool XConstraints::eraseConstraint ( idx_t idof )
{
  checkDofIndex ( idof );

  const idx_t  islave = slaveMap_[idof] - 1;

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


idx_t XConstraints::eraseConstraints ( const IdxVector& idofs )
{
  checkDofIndices ( idofs );

  const idx_t  n = idofs.size ();

  idx_t        k = 0;


  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  islave = slaveMap_[idofs[i]] - 1;

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


idx_t XConstraints::getConstraint

  ( double&           rval,
    const IdxVector&  jdofs,
    const Vector&     coeffs,
    idx_t             idof ) const

{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  n = jdofs.size ();

  MasterNode*  master;

  idx_t        islave;
  idx_t        j;


  islave = slaveMap_[idof] - 1;

  if ( islave < 0 )
  {
    return 0;
  }

  master = slaves_[islave].masterList;
  rval   = slaves_[islave].rvalue;

  for ( j = 0; master && j < n; master = master->next, j++ )
  {
    jdofs [j] = master->jdof;
    coeffs[j] = master->coeff;
  }

  return j;
}


//-----------------------------------------------------------------------
//   setRvalue
//-----------------------------------------------------------------------


void XConstraints::setRvalue

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  checkDofIndex ( idof );

  const idx_t   islave = slaveMap_[idof] - 1;
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


void XConstraints::setRvalues

  ( const IdxVector&  idofs,
    const Vector&     rvals )

{
  checkDofIndices ( idofs );

  const idx_t*  map = slaveMap_.addr ();
  const idx_t   n   = idofs    .size ();


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


void XConstraints::unsetRvalues ( const IdxVector& idofs )
{
  checkDofIndices ( idofs );

  const idx_t*  map = slaveMap_.addr ();
  const idx_t   n   = idofs    .size ();


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


double XConstraints::getRvalue ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return 0.0;
  }

  const idx_t  islave = slaveMap_[idof] - 1;

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


void XConstraints::getRvalues

  ( const Vector&     rvals,
    const IdxVector&  idofs ) const

{
  const idx_t*  map = slaveMap_.addr ();
  const idx_t   n   = idofs    .size ();


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


void XConstraints::getRvalues ( const SparseMatrix& cmat ) const
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

    idx_t  islave = slaveMap_[idof] - 1;

    if ( islave >= 0 )
    {
      values[i] = slaves_[islave].rvalue;
    }
  }
}


//-----------------------------------------------------------------------
//   setSlaveDofs
//-----------------------------------------------------------------------


void XConstraints::setSlaveDofs

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
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      vec[idof]   = slaves[i].rvalue * rscale;
    }
  }
}


//-----------------------------------------------------------------------
//   evalSlaveDofs
//-----------------------------------------------------------------------


void XConstraints::evalSlaveDofs

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
      master      = slaves[i].masterList;

      while ( master )
      {
        v[idof] += master->coeff * v[master->jdof];
        master   = master->next;
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      vec[idof]   = slaves[i].rvalue * rscale;
      master      = slaves[i].masterList;

      while ( master )
      {
        vec[idof] += master->coeff * vec[master->jdof];
        master     = master->next;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   evalMasterDofs
//-----------------------------------------------------------------------


void XConstraints::evalMasterDofs ( const Vector& vec ) const
{
  const MasterNode*  master = 0;
  const SlaveNode*   slaves = slaves_.addr ();
  const idx_t          n      = slaves_.size ();


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      master      = slaves[i].masterList;

      while ( master )
      {
        v[master->jdof] += master->coeff * v[idof];
        master           = master->next;
      }

      v[idof] = 0.0;
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  idof = (idx_t) (slaves[i].idof >> 1_uidx);

      master      = slaves[i].masterList;

      while ( master )
      {
        vec[master->jdof] += master->coeff * vec[idof];
        master             = master->next;
      }

      vec[idof] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalueMask
//-----------------------------------------------------------------------


BoolVector XConstraints::getRvalueMask () const
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


void XConstraints::getRvalueMask

  ( const BoolVector&  rmask,
    const IdxVector&   idofs ) const

{
  const idx_t*  map = slaveMap_.addr ();
  const idx_t   n   = idofs    .size ();


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


SparseMatrix XConstraints::toMatrix () const
{
  const idx_t  dofCount   = dofs_->dofCount ();
  const idx_t  slaveCount = slaves_.size    ();

  IdxVector    offsets ( slaveCount + 1 );
  IdxVector    indices ( slaveCount + masterCount_ );
  Vector       values  ( slaveCount + masterCount_ );

  MasterNode*  master;

  idx_t        j = 0;


  for ( idx_t islave = 0; islave < slaveCount; islave++ )
  {
    const SlaveNode&  s = slaves_[islave];

    offsets[islave] = j;

    indices[j] = (idx_t) (s.idof >> 1_uidx);
    values [j] = s.rvalue;
    master     = s.masterList;

    for ( j++; master; master = master->next, j++ )
    {
      indices[j] = master->jdof;
      values [j] = master->coeff;
    }
  }

  offsets[slaveCount] = j;

  return SparseMatrix ( jem::shape( slaveCount, dofCount ),
                        offsets, indices, values );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void XConstraints::printTo ( PrintWriter& out ) const
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
    const SlaveNode&  slave  = slaves_[islave];
    MasterNode*       master = slave.masterList;

    idx_t  idof = (idx_t) (slave.idof >> 1_uidx);
    idx_t  j    = 0;

    dofs.decodeDofIndex ( iitem, itype, idof );

    idx_t  itemID = imap.getItemID ( iitem );

    if ( islave > 0 )
    {
      print ( out, endl );
    }

    print ( out, typeNames[itype], '[', itemID, "] = " );

    if ( ! master || (slave.idof & JEM_UIDX_C(0x1)) )
    {
      print ( out, slave.rvalue );

      j = 1;
    }

    for ( ; master; master = master->next, j++ )
    {
      double  coeff = master->coeff;

      idof = master->jdof;

      dofs.decodeDofIndex ( iitem, itype, idof );

      itemID = imap.getItemID ( iitem );

      if ( j > 0 )
      {
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


void XConstraints::newDofCount ()
{
  const idx_t  oldCount = slaveMap_.size ();

  if ( dofCount_ == oldCount )
  {
    return;
  }

  if ( dofCount_ < oldCount )
  {
    MasterNode*  master;
    MasterNode*  next;

    idx_t        islave = 0;

    while ( islave < slaves_.size() )
    {
      idx_t  idof = (idx_t) (slaves_[islave].idof >> 1_uidx);

      if ( idof >= dofCount_ )
      {
        freeSlave_ ( islave );
      }
      else
      {
        master = slaves_[islave].masterList;

        while ( master )
        {
          idx_t  jdof = master->jdof;

          next = master->next;

          if ( jdof >= dofCount_ )
          {
            freeMaster_ ( master );
          }

          master = next;
        }

        islave++;
      }
    }
  }

  newDofCount_ ();
}


//-----------------------------------------------------------------------
//   newDofOrder
//-----------------------------------------------------------------------


void XConstraints::newDofOrder ( const Reordering& reord )
{
  const idx_t* JEM_RESTRICT  fmap = reord.fmapPtr ();

  MasterNode*  master;
  MasterNode*  next;

  idx_t        islave = 0;


  // Stage 1: delete all slave and master nodes associated with
  // DOFs that have been deleted.

  while ( islave < slaves_.size() )
  {
    idx_t  idof = (idx_t) (slaves_[islave].idof >> 1_uidx);

    if ( fmap[idof] < 0 )
    {
      freeSlave_ ( islave );
    }
    else
    {
      master = slaves_[islave].masterList;

      while ( master )
      {
        next = master->next;

        if ( fmap[master->jdof] < 0 )
        {
          freeMaster_ ( master );
        }

        master = next;
      }

      islave++;
    }
  }

  // Stage 2: update the DOF indices and the map arrays.

  newDofCount_ ();

  slaveMap_  = 0;
  masterMap_ = (MasterNode*) 0;

  for ( islave = 0; islave < slaves_.size(); islave++ )
  {
    SlaveNode&    s = slaves_[islave];
    idx_t     idof  = (idx_t) (s.idof >> 1_uidx);

    idof            = fmap[idof];
    s.idof          = ((uidx_t) idof << 1_uidx) |
                       (s.idof & JEM_UIDX_C(0x1));
    slaveMap_[idof] = islave + 1;

    for ( master = s.masterList; master; master = master->next )
    {
      idx_t  jdof  = fmap[master->jdof];

      master->jdof = jdof;

      if ( ! master->down )
      {
        JEM_ASSERT ( masterMap_[jdof] == 0 );

        masterMap_[jdof] = master;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   freeAll_
//-----------------------------------------------------------------------


void XConstraints::freeAll_ ()
{
  MasterNode*  master;
  MasterNode*  next;

  for ( idx_t islave = 0; islave < slaves_.size(); islave++ )
  {
    master = slaves_[islave].masterList;

    while ( master )
    {
      next   = master->next;

      allocator_.dealloc ( master );

      master = next;
    }
  }
}


//-----------------------------------------------------------------------
//   newSlave_
//-----------------------------------------------------------------------


idx_t XConstraints::newSlave_

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  JEM_ASSERT2 ( slaveMap_[idof] <= 0,
                "DOF already has a constraint" );

  const idx_t   islave = slaves_.size ();
  const uidx_t  umask  = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);

  SlaveNode     slave;


  slave.idof       = ((uidx_t) idof << 1_uidx) | umask;
  slave.rvalue     = rval;
  slave.masterList = 0;

  slaves_.pushBack ( std::move( slave ) );

  slaveMap_[idof]  = islave + 1;

  return islave;
}


idx_t XConstraints::newSlave_ ( const SlaveNode& orig )
{
  const idx_t  idof   = (idx_t) (orig.idof >> 1_uidx);
  const idx_t  islave = slaves_.size ();

  SlaveNode    copy;

  copy.idof       = orig.idof;
  copy.rvalue     = orig.rvalue;
  copy.masterList = 0;

  slaves_.pushBack ( std::move( copy ) );

  slaveMap_[idof] = islave + 1;

  if ( orig.masterList )
  {
    MasterNode*  master = orig.masterList;

    do
    {
      newMaster_ ( islave, master->jdof, master->coeff );

      master = master->next;
    }
    while ( master );
  }

  return islave;
}


//-----------------------------------------------------------------------
//   freeSlave_
//-----------------------------------------------------------------------


void XConstraints::freeSlave_ ( idx_t islave )
{
  SlaveNode&   slave = slaves_[islave];

  MasterNode*  master;
  MasterNode*  next;

  idx_t        jslave;
  idx_t        idof;
  idx_t        jdof;


  master = slave.masterList;

  while ( master )
  {
    jdof = master->jdof;
    next = master->next;

    if ( masterMap_[jdof] == master )
    {
      masterMap_[jdof] = master->up;
    }
    if ( master->up )
    {
      master->up->down = master->down;
    }
    if ( master->down )
    {
      master->down->up = master->up;
    }

    allocator_.dealloc ( master );
    masterCount_--;

    master = next;
  }

  idof            = (idx_t) (slave.idof >> 1_uidx);
  slaveMap_[idof] = 0;
  jslave          = slaves_.size() - 1;

  if ( islave != jslave )
  {
    slave = slaves_[jslave];

    for ( master = slave.masterList; master; master = master->next )
    {
      master->islave = islave;
    }

    idof            = (idx_t) (slave.idof >> 1_uidx);
    slaveMap_[idof] = islave + 1;
  }

  slaves_.popBack ();
}


//-----------------------------------------------------------------------
//   newMaster_
//-----------------------------------------------------------------------


XConstraints::MasterNode* XConstraints::newMaster_

  ( idx_t   islave,
    idx_t   jdof,
    double  coeff )

{
  SlaveNode&   slave  = slaves_[islave];
  MasterNode*  master = (MasterNode*) allocator_.alloc ();


  master->islave = islave;
  master->jdof   = jdof;
  master->coeff  = coeff;
  master->next   = slave.masterList;
  master->prev   = 0;

  if ( master->next )
  {
    master->next->prev = master;
  }

  slave.masterList = master;
  master->up       = masterMap_[jdof];
  master->down     = 0;
  masterMap_[jdof] = master;

  if ( master->up )
  {
    master->up->down = master;
  }

  masterCount_++;

  return master;
}


//-----------------------------------------------------------------------
//   freeMaster_
//-----------------------------------------------------------------------


void XConstraints::freeMaster_ ( MasterNode* master )
{
  JEM_ASSERT2 ( masterCount_ > 0, "no master DOFs" );

  SlaveNode&   slave = slaves_[master->islave];
  const idx_t  jdof  = master->jdof;


  if ( slave.masterList == master )
  {
    slave.masterList = master->next;
  }
  if ( master->next )
  {
    master->next->prev = master->prev;
  }
  if ( master->prev )
  {
    master->prev->next = master->next;
  }

  if ( masterMap_[jdof] == master )
  {
    masterMap_[jdof] = master->up;
  }
  if ( master->up )
  {
    master->up->down = master->down;
  }
  if ( master->down )
  {
    master->down->up = master->up;
  }

  allocator_.dealloc ( master );
  masterCount_--;
}


//-----------------------------------------------------------------------
//   addConstraint_
//-----------------------------------------------------------------------


void XConstraints::addConstraint_

  ( idx_t          idof,
    double         rval,
    bool           rmask,
    const idx_t*   jdofs,
    const double*  coeffs,
    idx_t          jcount )

{
  idx_t*       map = slaveMap_.addr ();
  Scratch_&    s   = scratch_;

  MasterNode*  master;
  MasterNode*  up;
  SlaveNode*   slave;

  double       coeff;
  double       scale;

  uidx_t       umask;
  idx_t        islave;
  idx_t        jdof;
  idx_t        j, k, n;


  umask     = rmask ? JEM_UIDX_C(0x1) : JEM_UIDX_C(0x0);
  map[idof] = 0 - 1;

  s.jdofs .clear ();
  s.coeffs.clear ();

  s.jdofs .pushBack ( idof );
  s.coeffs.pushBack ( -1.0 );

  for ( j = 0; j < jcount; j++ )
  {
    jdof   = jdofs [j];
    coeff  = coeffs[j];
    islave = map[jdof] - 1;

    if ( islave < 0 )
    {
      k = -islave - 2;

      if ( k < 0 )
      {
        map[jdof] = -1 - s.jdofs.size ();

        s.jdofs .pushBack ( jdof  );
        s.coeffs.pushBack ( coeff );
      }
      else
      {
        s.coeffs[k] += coeff;
      }
    }
    else
    {
      slave  = slaves_.addr ( islave );

      rval  += slave->rvalue * coeff;
      umask |= (slave->idof & JEM_UIDX_C(0x1));
      master = slave->masterList;

      for ( ; master; master = master->next )
      {
        jdof = master->jdof;
        k    = -map[jdof] - 1;

        if ( k < 0 )
        {
          map[jdof] = -1 - s.jdofs.size ();

          s.jdofs .pushBack ( jdof );
          s.coeffs.pushBack ( coeff * master->coeff );
        }
        else
        {
          s.coeffs[k] += coeff * master->coeff;
        }
      }
    }
  }

  scale     = -s.coeffs[0];
  map[idof] =  0;
  n         =  s.jdofs.size ();

  if ( jem::isTiny( scale ) )
  {
    resolveError ( idof );
  }

  scale  = 1.0 / scale;
  rval  *= scale;
  islave = newSlave_ ( idof, rval, (umask != 0) );

  for ( j = 1; j < n; j++ )
  {
    jdof         = s.jdofs[j];
    s.coeffs[j] *= scale;
    map[jdof]    = 0;

    newMaster_ ( islave, jdof, s.coeffs[j] );
  }

  master = masterMap_[idof];

  while ( master )
  {
    up     = master->up;
    islave = master->islave;
    slave  = slaves_.addr ( islave );
    scale  = master->coeff;

    freeMaster_ ( master );

    slave->rvalue += scale * rval;
    slave->idof   |= umask;
    master         = slave->masterList;

    s.accu.clear ();

    for ( ; master; master = master->next )
    {
      map[master->jdof] = -1 - s.accu.size ();

      s.accu.pushBack ( master->coeff );
    }

    for ( j = 1; j < n; j++ )
    {
      jdof = s.jdofs[j];
      k    = -map[jdof] - 1;

      if ( k < 0 )
      {
        newMaster_ ( islave, jdof, 0.0 );

        map[jdof] = -1 - s.accu.size ();

        s.accu.pushBack ( scale * s.coeffs[j] );
      }
      else
      {
        s.accu[k] += scale * s.coeffs[j];
      }
    }

    master = slave->masterList;

    for ( ; master; master = master->next )
    {
      jdof          = master->jdof;
      master->coeff = s.accu[-map[jdof] - 1];
      map[jdof]     = 0;
    }

    master = up;
  }

  events.set ( NEW_STRUCT );
}


//-----------------------------------------------------------------------
//   fixSlaveMap_
//-----------------------------------------------------------------------


void XConstraints::fixSlaveMap_ ()
{
  idx_t*  map = slaveMap_.addr ();

  for ( idx_t idof = 0; idof < dofCount_; idof++ )
  {
    if ( map[idof] < 0 )
    {
      map[idof] = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   newDofCount_
//-----------------------------------------------------------------------


void XConstraints::newDofCount_ ()
{
  const idx_t  oldCount = slaveMap_.size ();

  if      ( dofCount_ > oldCount )
  {
    const idx_t  n = dofCount_ - oldCount;

    slaveMap_ .pushBack (               0, n );
    masterMap_.pushBack ( (MasterNode*) 0, n );
  }
  else if ( dofCount_ < oldCount )
  {
    const idx_t  n = oldCount - dofCount_;

    slaveMap_ .popBack ( n );
    masterMap_.popBack ( n );
  }

  JEM_ASSERT ( slaveMap_.size() == masterMap_.size() );

  events.set ( NEW_STRUCT );
}


JIVE_END_PACKAGE( util )
