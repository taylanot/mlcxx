
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
#include <jem/io/PrintWriter.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Reordering.h>
#include "CConstraints.h"


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class CConstraints
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CConstraints::CConstraints ( const Ref<DofSpace>& dofs ) :

  Super ( dofs )

{
  slaveCount_ = 0;
  maxMasters_ = 0;

  conOffsets_.resize ( 1 );
  dofConMap_ .resize ( dofCount_ );

  conOffsets_[0] = 0;
  dofConMap_     = -1_idx;
}


CConstraints::CConstraints

  ( const Ref<DofSpace>&  dofs,
    const SparseMatrix&   cmat,
    const BoolVector&     rmask ) :

    Super ( dofs )

{
  JEM_ASSERT2 ( cmat.size(0) == rmask.size() &&
                cmat.size(1) == dofCount_,
                "constraints matrix has an invalid shape" );

  slaveCount_ = cmat.size (0);
  maxMasters_ = 0;

  conOffsets_.ref ( cmat.getRowOffsets()    );
  dofIndices_.ref ( cmat.getColumnIndices() );
  conCoeffs_ .ref ( cmat.getValues()        );
  rvalueMask_.ref ( rmask );

  if ( ! conOffsets_.isContiguous() )
  {
    conOffsets_.ref ( conOffsets_.clone() );
  }

  if ( ! dofIndices_.isContiguous() )
  {
    dofIndices_.ref ( dofIndices_.clone() );
  }

  if ( ! conCoeffs_.isContiguous() )
  {
    conCoeffs_.ref  ( conCoeffs_ .clone() );
  }

  if ( ! rvalueMask_.isContiguous() )
  {
    rvalueMask_.ref ( rvalueMask_.clone() );
  }

  initDofConMap_ ();
}


CConstraints::CConstraints ( const Self& rhs ) :

  Super ( rhs )

{
  const idx_t  m =     rhs.conOffsets_.size ();
  const idx_t  n = m + rhs.dofIndices_.size ();
  const idx_t  p = n + rhs.dofConMap_ .size ();

  IdxVector    ibuf ( p );

  conOffsets_.ref ( ibuf[slice(BEGIN,m)] );
  dofIndices_.ref ( ibuf[slice(m,n)] );
  dofConMap_ .ref ( ibuf[slice(n,END)] );

  slaveCount_ = rhs.slaveCount_;
  maxMasters_ = rhs.maxMasters_;
  conOffsets_ = rhs.conOffsets_;
  dofIndices_ = rhs.dofIndices_;
  dofConMap_  = rhs.dofConMap_;

  conCoeffs_ .ref ( rhs.conCoeffs_ .clone() );
  rvalueMask_.ref ( rhs.rvalueMask_.clone() );
}


CConstraints::~CConstraints ()
{}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<ConstraintsImp> CConstraints::clone () const
{
  return jem::newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void CConstraints::clear ()
{
  conOffsets_.ref ( IdxVector( 1 ) );
  dofIndices_.ref ( IdxVector()    );
  conCoeffs_ .ref ( Vector()       );
  rvalueMask_.ref ( BoolVector()   );

  events.set ( NEW_STRUCT );

  slaveCount_    =  0;
  maxMasters_    =  0;
  conOffsets_[0] =  0;
  dofConMap_     = -1;
}


//-----------------------------------------------------------------------
//   slaveDofCount
//-----------------------------------------------------------------------


idx_t CConstraints::slaveDofCount () const
{
  return slaveCount_;
}


//-----------------------------------------------------------------------
//   masterDofCount
//-----------------------------------------------------------------------


idx_t CConstraints::masterDofCount () const
{
  return (dofIndices_.size() - slaveCount_);
}


idx_t CConstraints::masterDofCount ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t  islave = dofConMap_[idof];

  if ( islave < 0 )
  {
    return 0;
  }
  else
  {
    return (conOffsets_[islave + 1] - conOffsets_[islave] - 1);
  }
}


//-----------------------------------------------------------------------
//   isSlaveDof
//-----------------------------------------------------------------------


bool CConstraints::isSlaveDof ( idx_t idof ) const
{
  if ( checkDofIndex( idof ) )
  {
    return (dofConMap_[idof] >= 0);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   isMasterDof
//-----------------------------------------------------------------------


bool CConstraints::isMasterDof ( idx_t idof ) const
{
  if ( checkDofIndex( idof ) )
  {
    return (dofConMap_[idof] < -1);
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   hasRvalue
//-----------------------------------------------------------------------


bool CConstraints::hasRvalue ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return false;
  }

  const idx_t  islave = dofConMap_[idof];

  if ( islave < 0 )
  {
    return false;
  }
  else
  {
    return rvalueMask_[islave];
  }
}


//-----------------------------------------------------------------------
//   getSlaveDofs
//-----------------------------------------------------------------------


idx_t CConstraints::getSlaveDofs ( const IdxVector& idofs ) const
{
  const idx_t*  offsets = conOffsets_.addr ();
  const idx_t*  indices = dofIndices_.addr ();

  const idx_t   n = jem::min ( slaveCount_, idofs.size() );

  for ( idx_t i = 0; i < n; i++ )
  {
    idofs[i] = indices[offsets[i]];
  }

  return n;
}


//-----------------------------------------------------------------------
//   getMasterDofs
//-----------------------------------------------------------------------


idx_t CConstraints::getMasterDofs

  ( const IdxVector&  jdofs,
    idx_t             idof ) const

{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const idx_t*  iptr;

  idx_t         islave;
  idx_t         i, n;


  islave = dofConMap_[idof];

  if ( islave < 0 )
  {
    return 0;
  }
  else
  {
    i    = conOffsets_[islave] + 1;
    n    = conOffsets_[islave  + 1] - i;
    iptr = dofIndices_.addr ( i );

    if ( jdofs.size() < n )
    {
      n = jdofs.size ();
    }

    for ( i = 0; i < n; i++ )
    {
      jdofs[i] = iptr[i];
    }

    return n;
  }
}


//-----------------------------------------------------------------------
//   getConstraint
//-----------------------------------------------------------------------


idx_t CConstraints::getConstraint

  ( double&           rval,
    const IdxVector&  jdofs,
    const Vector&     coeffs,
    idx_t             idof ) const

{
  if ( ! checkDofIndex( idof ) )
  {
    return 0;
  }

  const double*  JEM_RESTRICT  xptr;
  const idx_t*   JEM_RESTRICT  iptr;

  idx_t          islave;
  idx_t          i, n;


  islave = dofConMap_[idof];

  if ( islave < 0 )
  {
    return 0;
  }
  else
  {
    i    = conOffsets_[islave];
    rval = conCoeffs_[i];

    i++;

    n    = conOffsets_[islave + 1] - i;
    iptr = dofIndices_.addr ( i );
    xptr = conCoeffs_ .addr ( i );

    if ( jdofs.size() < n )
    {
      n = jdofs.size ();
    }

    for ( i = 0; i < n; i++ )
    {
      jdofs [i] = iptr[i];
      coeffs[i] = xptr[i];
    }

    return n;
  }
}


//-----------------------------------------------------------------------
//   setRvalue
//-----------------------------------------------------------------------


void CConstraints::setRvalue

  ( idx_t   idof,
    double  rval,
    bool    rmask )

{
  checkDofIndex ( idof );

  const idx_t  islave = dofConMap_[idof];

  if ( islave >= 0 )
  {
    conCoeffs_[conOffsets_[islave]] = rval;

    if ( rvalueMask_[islave] == rmask )
    {
      if ( rmask )
      {
        events.set ( NEW_RVALS );
      }
    }
    else
    {
      events.set ( NEW_STRUCT );

      rvalueMask_[islave] = rmask;
    }
  }
}


//-----------------------------------------------------------------------
//   setRvalues
//-----------------------------------------------------------------------


void CConstraints::setRvalues

  ( const IdxVector&  idofs,
    const Vector&     rvals )

{
  checkDofIndices ( idofs );

  const idx_t*  offsets = conOffsets_.addr ();
  const idx_t*  dofMap  = dofConMap_ .addr ();
  double*       coeffs  = conCoeffs_ .addr ();
  bool*         rmask   = rvalueMask_.addr ();

  const idx_t   n = idofs.size ();

  idx_t         islave;
  idx_t         i;


  for ( i = 0; i < n; i++ )
  {
    islave = dofMap[idofs[i]];

    if ( islave >= 0 )
    {
      coeffs[offsets[islave]] = rvals[i];

      if ( rmask[islave] )
      {
        events.set ( NEW_RVALS );
      }
      else
      {
        events.set ( NEW_STRUCT );

        rmask[islave] = true;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   unsetRvalues
//-----------------------------------------------------------------------


void CConstraints::unsetRvalues ( const IdxVector& idofs )
{
  checkDofIndices ( idofs );

  const idx_t*  offsets = conOffsets_.addr ();
  const idx_t*  dofMap  = dofConMap_ .addr ();
  double*       coeffs  = conCoeffs_ .addr ();
  bool*         rmask   = rvalueMask_.addr ();

  const idx_t   n = idofs.size ();

  idx_t         islave;
  idx_t         i;


  for ( i = 0; i < n; i++ )
  {
    islave = dofMap[idofs[i]];

    if ( islave >= 0 && rmask[islave] )
    {
      coeffs[offsets[islave]] = 0.0;

      rmask[islave] = false;

      events.set ( NEW_STRUCT );
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalue
//-----------------------------------------------------------------------


double CConstraints::getRvalue ( idx_t idof ) const
{
  if ( ! checkDofIndex( idof ) )
  {
    return 0.0;
  }

  const idx_t  islave = dofConMap_[idof];

  if ( islave < 0 )
  {
    return 0.0;
  }
  else
  {
    return conCoeffs_[conOffsets_[islave]];
  }
}


//-----------------------------------------------------------------------
//   getRvalues (with array args)
//-----------------------------------------------------------------------


void CConstraints::getRvalues

  ( const Vector&     rvals,
    const IdxVector&  idofs ) const

{
  const idx_t*   offsets = conOffsets_.addr ();
  const idx_t*   dofMap  = dofConMap_ .addr ();
  const double*  coeffs  = conCoeffs_ .addr ();

  const idx_t    n = idofs.size ();

  idx_t          islave;
  idx_t          idof;
  idx_t          i;


  for ( i = 0; i < n; i++ )
  {
    idof = idofs[i];

    if ( ! checkDofIndex( idof ) )
    {
      continue;
    }

    islave = dofMap[idof];

    if ( islave >= 0 )
    {
      rvals[i] = coeffs[offsets[islave]];
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalues (with a SparseMatrix)
//-----------------------------------------------------------------------


void CConstraints::getRvalues ( const SparseMatrix& cmat ) const
{
  if ( cmat.getValues().addr() == conCoeffs_.addr() )
  {
    return;
  }

  IdxVector    offsets  = cmat.getRowOffsets    ();
  IdxVector    indices  = cmat.getColumnIndices ();
  Vector       values   = cmat.getValues        ();

  const idx_t  lastDof  = dofCount_ - 1;
  const idx_t  rowCount = cmat.size (0);

  idx_t        islave;
  idx_t        irow;
  idx_t        idof;
  idx_t        i, n;


  for ( irow = 0; irow < rowCount; irow++ )
  {
    i = offsets[irow];
    n = offsets[irow + 1];

    if ( i >= n )
    {
      continue;
    }

    idof = indices[i];

    if ( idof < 0 || idof > lastDof )
    {
      continue;
    }

    islave = dofConMap_[idof];

    if ( islave >= 0 )
    {
      values[i] = conCoeffs_[conOffsets_[islave]];
    }
  }
}


//-----------------------------------------------------------------------
//   getRvalueMask
//-----------------------------------------------------------------------


BoolVector CConstraints::getRvalueMask () const
{
  return rvalueMask_;
}


void CConstraints::getRvalueMask

  ( const BoolVector&  rmask,
    const IdxVector&   idofs ) const

{
  const idx_t*  dofMap = dofConMap_ .addr ();
  const bool*   myMask = rvalueMask_.addr ();

  const idx_t   n = idofs.size ();

  idx_t         islave;
  idx_t         idof;
  idx_t         i;


  for ( i = 0; i < n; i++ )
  {
    idof = idofs[i];

    if ( ! checkDofIndex( idof ) )
    {
      continue;
    }

    islave = dofMap[idof];

    if ( islave >= 0 )
    {
      rmask[i] = myMask[islave];
    }
  }
}


//-----------------------------------------------------------------------
//   setSlaveDofs
//-----------------------------------------------------------------------


void CConstraints::setSlaveDofs

  ( const Vector&  vec,
    double         rscale ) const

{
  const double*  JEM_RESTRICT  coeffs  = conCoeffs_ .addr ();
  const idx_t*   JEM_RESTRICT  offsets = conOffsets_.addr ();
  const idx_t*   JEM_RESTRICT  indices = dofIndices_.addr ();

  const idx_t  n = slaveCount_;

  idx_t        i, j;


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

JEM_IVDEP

    for ( i = 0; i < n; i++ )
    {
      j             = offsets[i];
      v[indices[j]] = coeffs [j] * rscale;
    }
  }
  else
  {

JEM_IVDEP

    for ( i = 0; i < n; i++ )
    {
      j               = offsets[i];
      vec[indices[j]] = coeffs [j] * rscale;
    }
  }
}


//-----------------------------------------------------------------------
//   evalSlaveDofs
//-----------------------------------------------------------------------


void CConstraints::evalSlaveDofs

  ( const Vector&  vec,
    double         rscale ) const

{
  if ( maxMasters_ == 0 )
  {
    Self::setSlaveDofs ( vec, rscale ); return;
  }

  const double*  JEM_RESTRICT  coeffs  = conCoeffs_ .addr ();
  const idx_t*   JEM_RESTRICT  offsets = conOffsets_.addr ();
  const idx_t*   JEM_RESTRICT  indices = dofIndices_.addr ();

  const idx_t  n = slaveCount_;

  double       r;

  idx_t        idof;
  idx_t        i, j, k;


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

    for ( i = 0; i < n; i++ )
    {
      j    = offsets[i];
      k    = offsets[i + 1];
      idof = indices[j];
      r    = coeffs [j] * rscale;

      for ( j++; j < k; j++ )
      {
        r += coeffs[j] * v[indices[j]];
      }

      v[idof] = r;
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      j    = offsets[i];
      k    = offsets[i + 1];
      idof = indices[j];
      r    = coeffs [j] * rscale;

      for ( j++; j < k; j++ )
      {
        r += coeffs[j] * vec[indices[j]];
      }

      vec[idof] = r;
    }
  }
}


//-----------------------------------------------------------------------
//   evalMasterDofs
//-----------------------------------------------------------------------


void CConstraints::evalMasterDofs ( const Vector& vec ) const
{
  if ( maxMasters_ == 0 )
  {
    Self::setSlaveDofs ( vec, 0.0 ); return;
  }

  const double*  JEM_RESTRICT  coeffs  = conCoeffs_ .addr ();
  const idx_t*   JEM_RESTRICT  offsets = conOffsets_.addr ();
  const idx_t*   JEM_RESTRICT  indices = dofIndices_.addr ();

  const idx_t  n = slaveCount_;

  double       r;

  idx_t        idof;
  idx_t        i, j, k;


  checkSize ( vec );

  if ( vec.isContiguous() )
  {
    double* JEM_RESTRICT  v = vec.addr ();

    for ( i = 0; i < n; i++ )
    {
      j    = offsets[i];
      k    = offsets[i + 1];
      idof = indices[j];
      r    = v[idof];

JEM_IVDEP

      for ( j++; j < k; j++ )
      {
        v[indices[j]] += r * coeffs[j];
      }

      v[idof] = 0.0;
    }
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      j    = offsets[i];
      k    = offsets[i + 1];
      idof = indices[j];
      r    = vec[idof];

JEM_IVDEP

      for ( j++; j < k; j++ )
      {
        vec[indices[j]] += r * coeffs[j];
      }

      vec[idof] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   toMatrix
//-----------------------------------------------------------------------


SparseMatrix CConstraints::toMatrix () const
{
  const idx_t  dofCount = dofs_->dofCount ();

  return SparseMatrix ( jem::shape ( slaveCount_, dofCount ),
                        conOffsets_, dofIndices_, conCoeffs_ );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void CConstraints::printTo ( PrintWriter& out ) const
{
  using jem::io::endl;

  const DofSpace&  dofs      = * dofs_;
  const ItemSet&   items     = * dofs .getItems   ();
  const ItemMap&   imap      = * items.getItemMap ();

  StringVector     typeNames = dofs.getTypeNames  ();

  idx_t            iitem;
  idx_t            itype;


  for ( idx_t islave = 0; islave < slaveCount_; islave++ )
  {
    idx_t  i    = conOffsets_[islave];
    idx_t  n    = conOffsets_[islave + 1];
    idx_t  idof = dofIndices_[i];
    idx_t  j    = 0;

    dofs.decodeDofIndex ( iitem, itype, idof );

    idx_t   itemID = imap.getItemID ( iitem );
    double  value  = conCoeffs_[i];

    if ( islave > 0 )
    {
      print ( out, endl );
    }

    print ( out, typeNames[itype], '[', itemID, "] = " );

    i++;

    if ( i == n || rvalueMask_[islave] )
    {
      print ( out, value );

      j = 1;
    }

    for ( ; i < n; i++, j++ )
    {
      idof = dofIndices_[i];

      dofs.decodeDofIndex ( iitem, itype, idof );

      itemID = imap.getItemID ( iitem );
      value  = conCoeffs_[i];

      if ( j > 0 )
      {
        if ( value < 0.0 )
        {
          value *= -1.0;

          print ( out, " - " );
        }
        else
        {
          print ( out, " + " );
        }
      }

      print ( out, value, " * ", typeNames[itype], '[', itemID, ']' );
    }

    print ( out, ';' );
  }
}


//-----------------------------------------------------------------------
//   newDofCount
//-----------------------------------------------------------------------


void CConstraints::newDofCount ()
{
  const idx_t  oldCount = dofConMap_.size ();
  const idx_t  newCount = dofCount_;

  if      ( newCount > oldCount )
  {
    dofConMap_.reshape ( newCount );

    dofConMap_[slice(oldCount,END)] = -1;

    events.set ( NEW_STRUCT );
  }
  else if ( newCount < oldCount )
  {
    idx_t*   offsets = conOffsets_.addr ();
    idx_t*   idofs   = dofIndices_.addr ();
    double*  coeffs  = conCoeffs_ .addr ();
    bool*    rmask   = rvalueMask_.addr ();

    idx_t    islave;
    idx_t    jslave;
    idx_t    i, j, n;


    jslave = j = 0;

    for ( islave = 0; islave < slaveCount_; islave++ )
    {
      i = offsets[islave];
      n = offsets[islave + 1];

      if ( idofs[i] < newCount )
      {
        idofs [j] = idofs [i];
        coeffs[j] = coeffs[i];
        j++;

        for ( i++; i < n; i++ )
        {
          if ( idofs[i] < newCount )
          {
            idofs [j] = idofs [i];
            coeffs[j] = coeffs[i];
            j++;
          }
        }

        rmask[jslave++] = rmask[islave];
        offsets[jslave] = j;
      }
    }

    slaveCount_ = jslave;

    events     .set     ( NEW_STRUCT );
    conOffsets_.reshape ( slaveCount_ + 1 );
    dofIndices_.reshape ( j );
    conCoeffs_ .reshape ( j );
    rvalueMask_.reshape ( slaveCount_ );

    initDofConMap_ ();
  }
}


//-----------------------------------------------------------------------
//   newDofOrder
//-----------------------------------------------------------------------


void CConstraints::newDofOrder ( const Reordering& reord )
{
  const idx_t*  JEM_RESTRICT  fmap    = reord.fmapPtr ();

  idx_t*        JEM_RESTRICT  offsets = conOffsets_.addr ();
  idx_t*        JEM_RESTRICT  idofs   = dofIndices_.addr ();
  double*       JEM_RESTRICT  coeffs  = conCoeffs_ .addr ();
  bool*         JEM_RESTRICT  rmask   = rvalueMask_.addr ();

  idx_t         islave;
  idx_t         jslave;
  idx_t         jdof;
  idx_t         i, j, n;


  jslave = j = 0;

  for ( islave = 0; islave < slaveCount_; islave++ )
  {
    i    = offsets[islave];
    n    = offsets[islave + 1];
    jdof = fmap   [idofs[i]];

    if ( jdof >= 0 )
    {
      idofs [j] = jdof;
      coeffs[j] = coeffs[i];
      j++;

      for ( i++; i < n; i++ )
      {
        jdof = fmap[idofs[i]];

        if ( jdof >= 0 )
        {
          idofs [j] = jdof;
          coeffs[j] = coeffs[i];
          j++;
        }
      }

      rmask[jslave++] = rmask[islave];
      offsets[jslave] = j;
    }
  }

  slaveCount_ = jslave;

  events     .set     ( NEW_STRUCT );
  conOffsets_.reshape ( slaveCount_ + 1 );
  dofIndices_.reshape ( j );
  conCoeffs_ .reshape ( j );
  rvalueMask_.reshape ( slaveCount_ );

  initDofConMap_ ();
}


//-----------------------------------------------------------------------
//   initDofConMap_
//-----------------------------------------------------------------------


void CConstraints::initDofConMap_ ()
{
  const idx_t*  JEM_RESTRICT  offsets = conOffsets_.addr ();
  const idx_t*  JEM_RESTRICT  idofs   = dofIndices_.addr ();
  idx_t*        JEM_RESTRICT  dofMap;

  idx_t         maxCount;
  idx_t         islave;
  idx_t         i, k, n;


  dofConMap_.resize ( dofCount_ );

  dofConMap_ = -1_idx;
  dofMap     = dofConMap_.addr ();
  maxCount   = 0;

  for ( islave = 0; islave < slaveCount_; islave++ )
  {
    i = offsets[islave];
    n = offsets[islave + 1];
    k = n - i - 1;

    JEM_ASSERT ( i < n );

    dofMap[idofs[i]] = islave;

    for ( i++; i < n; i++ )
    {
      dofMap[idofs[i]] = -2;
    }

    if ( k > maxCount )
    {
      maxCount = k;
    }
  }

  maxMasters_ = maxCount;
}


JIVE_END_PACKAGE( util )
