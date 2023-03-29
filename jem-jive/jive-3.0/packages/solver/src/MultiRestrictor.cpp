
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/solver/Names.h>
#include <jive/solver/MultiRestrictor.h>
#include <jive/solver/RestrictorFactory.h>


JEM_DEFINE_CLASS( jive::solver::MultiRestrictor );


JIVE_BEGIN_PACKAGE( solver )


using jive::util::sizeError;


//=======================================================================
//   class MultiRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MultiRestrictor::TYPE_NAME   = "Multi";

const int    MultiRestrictor::NEW_VALUES_ = 1 << 0;
const int    MultiRestrictor::NEW_STRUCT_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MultiRestrictor::MultiRestrictor ( const String& name ) :

  Super ( name )

{
  shape_   =  0;
  events_  = ~0x0;
  started_ =  0;
}


MultiRestrictor::~MultiRestrictor ()
{}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape MultiRestrictor::shape () const
{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  return shape_;
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void MultiRestrictor::start ()
{
  if ( ! started_ )
  {
    resetEvents ();
  }

  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    rtor->start ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void MultiRestrictor::finish ()
{
  if ( started_ )
  {
    started_--;
  }

  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    rtor->finish ();
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MultiRestrictor::update ()
{
  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    rtor->update ();
  }

  resetEvents ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void MultiRestrictor::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  using jem::slice;

  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  const idx_t  rowCount = shape_[0];
  const idx_t  colCount = shape_[1];
  idx_t        irow     = 0;


  if ( rhs.size() != colCount )
  {
    sizeError ( JEM_FUNC, "right-hand vector",
                rhs.size(), colCount );
  }

  if ( lhs.size() != rowCount )
  {
    sizeError ( JEM_FUNC, "left-hand vector",
                lhs.size(), rowCount );
  }

  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );
    const Shape      sh   = rtor->shape              ();
    idx_t            jrow = irow + sh[0];

    if ( sh[1] != colCount )
    {
      sizeError ( JEM_FUNC, rtor->getContext(),
                  sh[1], colCount );
    }

    JEM_PRECHECK ( jrow <= rowCount );

    rtor->matmul ( lhs[slice(irow,jrow)], rhs );

    irow = jrow;
  }

  JEM_PRECHECK ( irow == rowCount );
}


//-----------------------------------------------------------------------
//   transmul
//-----------------------------------------------------------------------


void MultiRestrictor::transmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  const idx_t  rowCount = shape_[0];
  const idx_t  colCount = shape_[1];
  idx_t        irow     = 0;


  if ( rhs.size() != rowCount )
  {
    sizeError ( JEM_FUNC, "right-hand vector",
                rhs.size(), rowCount );
  }

  if ( lhs.size() != colCount )
  {
    sizeError ( JEM_FUNC, "left-hand vector",
                lhs.size(), colCount );
  }

  lhs = 0.0;

  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );
    const Shape      sh   = rtor->shape              ();
    idx_t            jrow = irow + sh[0];

    if ( sh[1] != colCount )
    {
      sizeError ( JEM_FUNC, rtor->getContext(),
                  sh[1], colCount );
    }

    JEM_PRECHECK ( jrow <= rowCount );

    rtor->transmul ( scratch_, rhs[slice(irow,jrow)] );

    lhs += scratch_;
    irow = jrow;
  }

  JEM_PRECHECK ( irow == rowCount );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MultiRestrictor::configure ( const Properties& props )
{
  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    rtor->configure ( props );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MultiRestrictor::getConfig ( const Properties& props ) const
{
  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    rtor->getConfig ( props );
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* MultiRestrictor::getExtByID ( ExtensionID extID ) const
{
  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    if ( ! rtor->getExtByID( extID ) )
    {
      return nullptr;
    }
  }

  if ( extID == LocalRestrictor::ID )
  {
    LocalRestrictor*  ext = const_cast<Self*> ( this );

    return ext;
  }

  if ( extID == SparseMatrixExt::ID )
  {
    SparseMatrixExt*  ext = const_cast<Self*> ( this );

    return ext;
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void MultiRestrictor::resetEvents ()
{
  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );

    rtor->resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   addRestrictor
//-----------------------------------------------------------------------


void MultiRestrictor::addRestrictor ( const Ref<Restrictor>& rtor )
{
  JEM_PRECHECK ( rtor );

  rtors_.pushBack ( rtor );

  connect ( rtor->newStructEvent, this, &Self::structChanged_ );
  connect ( rtor->newValuesEvent, this, &Self::valuesChanged_ );

  rtor->resetEvents ();
  structChanged_    ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Restrictor> MultiRestrictor::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::newInstance;
  using jive::util::joinNames;

  Properties    myProps = props.findProps   ( name );
  Properties    myConf  = conf .makeProps   ( name );
  Ref<Self>     multi   = newInstance<Self> ( name );

  StringVector  names;


  myProps.find ( names, PropNames::RESTRICTORS );
  myConf .set  ( PropNames::RESTRICTORS, names );

  for ( idx_t i = 0; i < names.size(); i++ )
  {
    multi->addRestrictor (
      newRestrictor (
        joinNames ( name, names[i] ),
        conf, props, params, globdat
      )
    );
  }

  return multi;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MultiRestrictor::declare ()
{
  RestrictorFactory::declare ( TYPE_NAME,  & Self::makeNew );
  RestrictorFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   getBasis
//-----------------------------------------------------------------------


void MultiRestrictor::getBasis

  ( const Vector&  v,
    idx_t          j ) const

{
  using jive::util::indexError;

  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  if ( j < 0 || j >= shape_[0] )
  {
    indexError ( JEM_FUNC, "basis vector", j,        shape_[0] );
  }

  if ( v.size() != shape_[1] )
  {
    sizeError  ( JEM_FUNC, "basis vector", v.size(), shape_[1] );
  }

  idx_t  irow = 0;

  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>   rtor = rtors_.getAs<Restrictor>         ( i );
    LocalRestrictor*  ext  = rtor->getExtension<LocalRestrictor> ();
    const Shape       sh   = rtor ->shape                        ();
    idx_t             jrow = irow + sh[0];

    if ( (j >= irow) && (j < jrow) )
    {
      ext->getBasis ( v, j - irow );
      break;
    }

    irow = jrow;
  }
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


SparseStruct MultiRestrictor::getStructure () const
{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  idx_t      irow       = 0;
  idx_t      nnz        = 0;

  IdxVector  rowOffsets ( shape_[0] + 1 );
  IdxVector  colIndices;


  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>   rtor    = rtors_.getAs<Restrictor> ( i );
    SparseMatrixExt*  ext     =

      rtor->getExtension<SparseMatrixExt> ();

    SparseStruct      sps     = ext->getStructure    ();
    IdxVector         offsets = sps.getRowOffsets    ();
    IdxVector         indices = sps.getColumnIndices ();
    const Shape       sh      = rtor->shape          ();
    idx_t             nnext   = nnz + indices.size   ();
    idx_t             nmax    = colIndices.size      ();

    JEM_PRECHECK ( (irow + sh[0]) <= shape_[0] );

    if ( nnext > nmax )
    {
      idx_t  n = jem::max ( nnext, nmax + nmax / 2 );

      colIndices.reshape ( n );
    }

    colIndices[slice(nnz,nnext)] = indices;

    for ( idx_t ii = 0; ii < sh[0]; ii++ )
    {
      rowOffsets[irow++] = nnz;
      nnz               += offsets[ii + 1] - offsets[ii];
    }

    JEM_PRECHECK ( nnz == nnext );

    irow += sh[0];
  }

  JEM_PRECHECK ( irow == shape_[0] );

  rowOffsets[irow] = nnz;

  colIndices.reshape ( nnz );

  return SparseStruct ( shape_, rowOffsets, colIndices );
}


//-----------------------------------------------------------------------
//   toSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix MultiRestrictor::toSparseMatrix () const
{
  if ( events_ & NEW_STRUCT_ )
  {
    const_cast<Self*> ( this ) -> update_ ();
  }

  idx_t      irow       = 0;
  idx_t      nnz        = 0;

  IdxVector  rowOffsets ( shape_[0] + 1 );
  IdxVector  colIndices;
  Vector     rtValues;


  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>   rtor    = rtors_.getAs<Restrictor> ( i );
    SparseMatrixExt*  ext     =

      rtor->getExtension<SparseMatrixExt> ();

    SparseMatrix      spm     = ext->toSparseMatrix  ();
    IdxVector         offsets = spm.getRowOffsets    ();
    IdxVector         indices = spm.getColumnIndices ();
    Vector            values  = spm.getValues        ();
    const Shape       sh      = rtor->shape          ();
    idx_t             nnext   = nnz + indices.size   ();
    idx_t             nmax    = colIndices.size      ();

    JEM_PRECHECK ( (irow + sh[0]) <= shape_[0] );

    if ( nnext > nmax )
    {
      idx_t  n = jem::max ( nnext, nmax + nmax / 2 );

      colIndices.reshape ( n );
      rtValues  .reshape ( n );
    }

    colIndices[slice(nnz,nnext)] = indices;
    rtValues  [slice(nnz,nnext)] = values;

    for ( idx_t ii = 0; ii < sh[0]; ii++ )
    {
      rowOffsets[irow++] = nnz;
      nnz               += offsets[ii + 1] - offsets[ii];
    }

    JEM_PRECHECK ( nnz == nnext );

    irow += sh[0];
  }

  JEM_PRECHECK ( irow == shape_[0] );

  rowOffsets[irow] = nnz;

  colIndices.reshape ( nnz );
  rtValues  .reshape ( nnz );

  return SparseMatrix ( shape_,     rowOffsets,
                        colIndices, rtValues );
}


//-----------------------------------------------------------------------
//   cloneSparseMatrix
//-----------------------------------------------------------------------


SparseMatrix MultiRestrictor::cloneSparseMatrix () const
{
  return Self::toSparseMatrix ();
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void MultiRestrictor::update_ ()
{
  shape_ = 0;

  for ( idx_t i = 0; i < rtors_.size(); i++ )
  {
    Ref<Restrictor>  rtor = rtors_.getAs<Restrictor> ( i );
    const Shape      sh   = rtor ->shape                ();

    shape_[0] += sh[0];

    if      ( ! shape_[1] )
    {
      shape_[1] = sh[1];
    }
    else if ( shape_[1] != sh[1] )
    {
      sizeError ( JEM_FUNC, rtor->getContext(),
                  sh[1], shape_[1] );
    }
  }

  scratch_.resize ( shape_[1] );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void MultiRestrictor::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void MultiRestrictor::structChanged_ ()
{
  setEvents_ ( NEW_STRUCT_ );

  shape_ = 0;

  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void MultiRestrictor::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "coarse grid restrictor changed while solving a "
      "linear system of equations"
    );
  }

  events_ |= events;
}


JIVE_END_PACKAGE( solver )
