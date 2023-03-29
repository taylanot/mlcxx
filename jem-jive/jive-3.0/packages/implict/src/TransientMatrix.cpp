
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


#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/implict/TransientMatrix.h>


JEM_DEFINE_CLASS( jive::implict::TransientMatrix );


JIVE_BEGIN_PACKAGE( implict )


using jem::equal;
using jem::isTiny;
using jem::newInstance;
using jem::IllegalInputException;
using jive::model::Actions;
using jive::model::ActionParams;


//=======================================================================
//   class TransientMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  TransientMatrix::U_MATRIX_  = 1 << 0;
const int  TransientMatrix::U_MATRIX0_ = 1 << 1;
const int  TransientMatrix::U_MATRIX1_ = 1 << 2;
const int  TransientMatrix::U_MATRIX2_ = 1 << 3;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TransientMatrix::TransientMatrix

  ( const String&      name,
    int                order,
    const Ref<Model>&  model,
    const Properties&  globdat ) :

    Super  ( name  ),
    model_ ( model )

{
  JEM_PRECHECK ( order >= 1 && order <= 2 && model );

  using jem::util::connect;
  using jive::algebra::SparseMatrixExt;

  Properties      params;

  SparseMatrixObj
    ::Traits      traits = SparseMatrixObj::SYMMETRIC;
  bool            sparse = true;


  updated_   = 0;
  coeffs_[0] = 1.0;
  coeffs_[1] = 0.0;
  coeffs_[2] = 0.0;

  for ( int i = 0; i <= order; i++ )
  {
    Ref<AbstractMatrix>  matrix;

    model->takeAction ( Actions::NEW_MATRIX[i], params, globdat );

    if ( ! params.find( matrix, ActionParams::MATRIX[i] ) )
    {
      String  ctx = model->getContext ();

      throw IllegalInputException (
        getContext     (),
        String::format (

          "failed to get matrix%d "
          "(%s ignored the action `%s\')",

          i, ctx, Actions::NEW_MATRIX[i]
        )
      );
    }

    matrices_[i] = matrix;

    params.erase ( ActionParams::MATRIX[i] );

    if ( ! matrix->hasExtension<SparseMatrixExt>() )
    {
      sparse = false;
    }

    if ( ! matrix->isSymmetric() )
    {
      traits = 0;
    }

    connect ( matrix->newValuesEvent,
              this, &Self::valuesChanged_ );
    connect ( matrix->newStructEvent,
              this, &Self::structChanged_ );

    matrix->resetEvents ();
  }

  if ( sparse )
  {
    transmat_ = newInstance<SparseMatrixObj> ( myName_, traits );
  }
}


TransientMatrix::~TransientMatrix ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void TransientMatrix::resetEvents ()
{
  for ( int i = 0; i < 3; i++ )
  {
    if ( matrices_[i] )
    {
      matrices_[i]->resetEvents ();
    }
  }
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape TransientMatrix::shape () const
{
  return matrices_[0]->shape ();
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void TransientMatrix::matmul

  ( const Vector&  lhs,
    const Vector&  rhs  ) const

{
  using jem::numeric::axpy;

  const Shape  sh = matrices_[0]->shape ();


  if ( vbuf_.size() != lhs.size() )
  {
    const_cast<Self*>( this )->vbuf_.resize ( lhs.size() );
  }

  lhs = 0.0;

  for ( int i = 0; i < 3; i++ )
  {
    if ( ! matrices_[i] || isTiny( coeffs_[i] ) )
    {
      continue;
    }

    if ( ! equal( sh, matrices_[i]->shape() ) )
    {
      shapeError_ ( i );
    }

    matrices_[i]->matmul ( vbuf_, rhs );

    axpy ( lhs, coeffs_[i], vbuf_ );
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool TransientMatrix::hasTrait ( const String& trait ) const
{
  bool  result = true;

  for ( int i = 0; i < 3; i++ )
  {
    if ( matrices_[i] && ! matrices_[i]->hasTrait( trait ) )
    {
      result = false;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


Ref<AbstractMatrix> TransientMatrix::getMatrix () const
{
  using jive::algebra::MPMatrixExt;
  using jive::algebra::MPMatrixObj;

  Ref<AbstractMatrix>  retval = transmat_;

  if ( ! retval )
  {
    retval = const_cast<Self*> ( this );
  }
  else
  {
    // Wrap the sparse matrix in an MP matrix if necessary.

    MPMatrixExt*  mpExt =

      matrices_[0]->getExtension<MPMatrixExt> ();

    if ( mpExt )
    {
      retval = newInstance<MPMatrixObj> ( retval,
                                          mpExt->getExchanger() );
    }
  }

  return retval;
}


AbstractMatrix* TransientMatrix::getMatrix ( int imat ) const
{
  AbstractMatrix*  retval = 0;

  if ( imat >= 0 && imat <= 2 )
  {
    retval = matrices_[imat].get ();
  }

  return retval;
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* TransientMatrix::getExtByID ( ExtensionID extID ) const
{
  using jive::algebra::MPMatrixExt;

  if ( extID == MPMatrixExt::ID )
  {
    return matrices_[0]->getExtByID ( extID );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void TransientMatrix::update

  ( const Vector&      fint,
    const Properties&  globdat )

{
  using jive::algebra::SparseMatrixExt;

  updateMatrix0 ( fint, globdat );
  updateMatrix1 (       globdat );
  updateMatrix2 (       globdat );

  if ( ! (updated_ & U_MATRIX_) && transmat_ )
  {
    const Shape  sh = matrices_[0]->shape ();

    // Reshape the transient matrix if required.

    if ( ! equal( sh, transmat_->shape() ) )
    {
      IdxVector  offsets ( sh[0] + 1 );
      IdxVector  indices;
      Vector     values;

      offsets = 0;

      transmat_->setMatrix (
        SparseMatrix ( sh, offsets, indices, values )
      );
    }

    // Assemble the transient matrix.

    transmat_->setToZero ();

    for ( int i = 0; i < 3; i++ )
    {
      if ( ! matrices_[i] || isTiny( coeffs_[i] ) )
      {
        continue;
      }

      SparseMatrix  smat = matrices_[i] ->

        getExtension<SparseMatrixExt>()->toSparseMatrix ();

      if ( smat.nonZeroCount() == 0 )
      {
        // Forget about it, even if the matrix has an invalid shape.
        continue;
      }

      if ( ! equal( sh, smat.shape() ) )
      {
        shapeError_ ( i );
      }

      transmat_->addMatrix ( smat, coeffs_[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   updateMatrix0
//-----------------------------------------------------------------------


AbstractMatrix* TransientMatrix::updateMatrix0

  ( const Vector&      fint,
    const Properties&  globdat )

{
  AbstractMatrix*  m0 = matrices_[0].get ();

  Properties       params;


  params.set ( ActionParams::INT_VECTOR, fint );

  if ( updated_ & U_MATRIX0_ )
  {
    model_->takeAction ( Actions::GET_INT_VECTOR, params, globdat );
  }
  else
  {
    model_->takeAction ( Actions::UPD_MATRIX0,    params, globdat );

    updated_ |= U_MATRIX0_;

    m0->resetEvents ();
  }

  return m0;
}


//-----------------------------------------------------------------------
//   updateMatrix1
//-----------------------------------------------------------------------


AbstractMatrix* TransientMatrix::updateMatrix1

  ( const Properties&  globdat )

{
  AbstractMatrix*  m1 = matrices_[1].get ();

  if ( m1 && ! (updated_ & U_MATRIX1_) )
  {
    Properties  params;

    model_->takeAction ( Actions::UPD_MATRIX1, params, globdat );

    updated_ |= U_MATRIX1_;

    m1->resetEvents ();
  }

  return m1;
}


//-----------------------------------------------------------------------
//   updateMatrix2
//-----------------------------------------------------------------------


AbstractMatrix* TransientMatrix::updateMatrix2

  ( const Properties&  globdat )

{
  AbstractMatrix*  m2 = matrices_[2].get ();

  if ( m2 && ! (updated_ & U_MATRIX2_) )
  {
    Properties  params;

    model_->takeAction ( Actions::UPD_MATRIX2, params, globdat );

    updated_ |= U_MATRIX2_;

    m2->resetEvents ();
  }

  return m2;
}


//-----------------------------------------------------------------------
//   setCoeffs
//-----------------------------------------------------------------------


void TransientMatrix::setCoeffs

  ( double  a,
    double  b,
    double  c )

{
  coeffs_[0] = a;
  coeffs_[1] = b;
  coeffs_[2] = c;

  updated_ &= ~U_MATRIX_;

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   shapeError_
//-----------------------------------------------------------------------


void TransientMatrix::shapeError_ ( int imat ) const
{
  const String  ctx = matrices_[imat]->getContext ();
  const Shape   sh  = matrices_[imat]->shape ();
  const Shape   sh0 = matrices_[0]   ->shape ();

  throw IllegalInputException (
    getContext     (),
    String::format (

      "%s has an invalid shape: (%d x %d); should be "
      "(%d x %d)",

      ctx, sh[0], sh[1], sh0[0], sh0[1]
    )
  );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void TransientMatrix::valuesChanged_ ()
{
  updated_ &= ~U_MATRIX_;

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void TransientMatrix::structChanged_ ()
{
  updated_ &= ~U_MATRIX_;

  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( implict )
