
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
#include <jem/base/Float.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/solver/Names.h>
#include <jive/solver/ConstrainerFactory.h>
#include <jive/solver/PointConstrainer.h>


JEM_DEFINE_CLASS( jive::solver::PointConstrainer       );
JEM_DEFINE_CLASS( jive::solver::DirectPointConstrainer );
JEM_DEFINE_CLASS( jive::solver::FilterPointConstrainer );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::System;
using jive::util::joinNames;
using jive::util::sizeError;
using jive::algebra::SparseMatrixExt;


//=======================================================================
//   class PointConstrainer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PointConstrainer::TYPE_NAME = "Point";
const char*  PointConstrainer::MODE_PROP = "mode";

const int    PointConstrainer::NEW_RVALUES_    = 1 << 0;
const int    PointConstrainer::NEW_SLAVE_DOFS_ = 1 << 1;
const int    PointConstrainer::NEW_MAT_VALUES_ = 1 << 2;
const int    PointConstrainer::NEW_MAT_STRUCT_ = 1 << 3;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PointConstrainer::PointConstrainer

  ( const String&         name,
    Ref<Constraints>      cons,
    Ref<AbstractMatrix>   input,
    Ref<SparseMatrixObj>  output ) :

    Super   (   name ),

    cons_   (   cons ),
    input_  (  input ),
    output_ ( output )

{
  using jem::util::connect;

  JEM_PRECHECK ( cons && input );

  if ( ! output_ )
  {
    SparseMatrixObj::Traits  traits = 0;

    if ( input->isSymmetric() )
    {
      traits |= SparseMatrixObj::SYMMETRIC;
    }

    output_ = newInstance<SparseMatrixObj>

      ( joinNames( myName_, PropNames::MATRIX ), traits );
  }

  events_   = ~0x0;
  warnFlag_ = false;

  connect ( cons_->newRvaluesEvent,
            this, & Self::rvaluesChanged_ );

  connect ( cons_->newStructEvent,
            this, & Self::slaveDofsChanged_ );

  connect ( input_->newValuesEvent,
            this, & Self::matValuesChanged_ );

  connect ( input_->newStructEvent,
            this, & Self::matStructChanged_ );

  if ( input != output )
  {
    // This is to make sure that the output matrix is updated
    // if it is modified externally

    connect ( output_->newValuesEvent,
              this, & Self::matValuesChanged_ );

    connect ( output_->newStructEvent,
              this, & Self::matStructChanged_ );
  }
}


PointConstrainer::~PointConstrainer ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void PointConstrainer::update ()
{
  idx_t  dofCount = dofMask_.size  ();
  idx_t  n        = input_ ->shape ()[0];

  idx_t  i, k;


  if ( n != dofCount )
  {
    if ( ! input_->isSquare() )
    {
      util::nonSquareMatrixError ( getContext(), input_->shape() );
    }

    dofCount = n;

    dofMask_.resize ( dofCount );
    rhs_    .resize ( dofCount );

    events_ |= (NEW_MAT_STRUCT_ | NEW_MAT_VALUES_ | NEW_SLAVE_DOFS_);
  }

  if ( events_ & NEW_SLAVE_DOFS_ )
  {
    IdxVector  idofs = cons_->getSlaveDofs ();

    n = idofs.size ();
    k = 0;

    for ( i = 0; i < n; i++ )
    {
      if ( idofs[i] < dofCount )
      {
        k++;
      }
    }

    slaveDofs_.resize ( k );
    rvalues_  .resize ( k );

    k = 0;

    for ( i = 0; i < n; i++ )
    {
      if ( idofs[i] < dofCount )
      {
        slaveDofs_[k++] = idofs[i];
      }
    }

    events_ |= NEW_RVALUES_;
  }

  if ( events_ & NEW_RVALUES_ )
  {
    cons_->getRvalues ( rvalues_, slaveDofs_ );

    if ( rvalues_.size() > 0 )
    {
      double  chksum = jem::sum ( rvalues_ );

      if ( jem::Float::isNaN( chksum ) )
      {
        throw jem::ArithmeticException (
          getContext (),
          "invalid constraints (NaN)"
        );
      }
    }
  }

  if ( events_ & (NEW_MAT_STRUCT_ | NEW_SLAVE_DOFS_) )
  {
    dofMask_             = true;
    dofMask_[slaveDofs_] = false;
  }

  if ( cons_->masterDofCount() > 0 && ! warnFlag_ )
  {
    print ( System::warn(), myName_,
            " : master dofs in constraints will be ignored.\n" );

    warnFlag_ = true;
  }

  doUpdate_ ();

  cons_ ->resetEvents ();
  input_->resetEvents ();

  if ( output_ != input_ )
  {
    output_->resetEvents ();
  }

  events_ = 0;
}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


bool PointConstrainer::updated () const
{
  return (events_ == 0);
}


//-----------------------------------------------------------------------
//   getRhs
//-----------------------------------------------------------------------


void PointConstrainer::getRhs

  ( const Vector&  g,
    const Vector&  f ) const

{
  JEM_PRECHECK ( updated() && g.size() == f.size() );

  const idx_t  dofCount = dofMask_.size ();


  if ( f.size() != dofCount )
  {
    sizeError ( getContext(), "rhs vector", f.size(), dofCount );
  }

  g  = rhs_;
  g += jem::where ( dofMask_, f, 0.0 );
}


//-----------------------------------------------------------------------
//   getLhs
//-----------------------------------------------------------------------


void PointConstrainer::getLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  JEM_PRECHECK ( updated() && x.size() == y.size() );

  const idx_t  dofCount = dofMask_.size ();

  if ( x.size() != dofCount )
  {
    sizeError ( getContext(), "lhs vector", x.size(), dofCount );
  }

  x             = y;
  x[slaveDofs_] = rvalues_;
}


//-----------------------------------------------------------------------
//   initLhs
//-----------------------------------------------------------------------


void PointConstrainer::initLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  JEM_PRECHECK ( updated() && x.size() == y.size() );

  const idx_t  dofCount = dofMask_.size ();

  if ( x.size() != dofCount )
  {
    sizeError ( getContext(), "lhs vector", x.size(), dofCount );
  }

  x             = y;
  x[slaveDofs_] = 0.0;
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* PointConstrainer::getConstraints () const
{
  return cons_.get ();
}


//-----------------------------------------------------------------------
//   getInputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* PointConstrainer::getInputMatrix () const
{
  return input_.get ();
}


//-----------------------------------------------------------------------
//   getOutputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* PointConstrainer::getOutputMatrix () const
{
  return output_.get ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Constrainer> PointConstrainer::makeNew

  ( const String&        myName,
    const Properties&    conf,
    const Properties&    props,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix )

{
  Ref<SparseMatrixObj>  smat =

    jem::dynamicCast<SparseMatrixObj> ( matrix );


  if ( smat )
  {
    Properties  myConf  = conf .makeProps ( myName );
    Properties  myProps = props.findProps ( myName );

    String      mode    = "Direct";


    myProps.find ( mode, MODE_PROP );
    myConf .set  ( MODE_PROP, mode );

    if ( mode.equalsIgnoreCase( "direct" ) )
    {
      return newInstance<DirectPointConstrainer>

        ( myName, cons, smat );
    }

    if ( mode.equalsIgnoreCase( "filter" ) )
    {
      return newInstance<FilterPointConstrainer> (
        myName, cons,
        smat, newInstance<SparseMatrixObj> (
          joinNames ( myName, PropNames::MATRIX )
        )
      );
    }

    myProps.propertyError (
      MODE_PROP,
      String::format (
        "illegal mode: `%s\'; should be `Direct\' or `Filter\'",
        mode
      )
    );
  }

  if ( matrix->hasExtension<SparseMatrixExt>() )
  {
    return newInstance<FilterPointConstrainer> (
      myName, cons,
      matrix
    );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void PointConstrainer::declare ()
{
  ConstrainerFactory::declare ( TYPE_NAME,  & makeNew );
  ConstrainerFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   rvaluesChanged_
//-----------------------------------------------------------------------


void PointConstrainer::rvaluesChanged_ ()
{
  events_ |= NEW_RVALUES_;
}


//-----------------------------------------------------------------------
//   slaveDofsChanged_
//-----------------------------------------------------------------------


void PointConstrainer::slaveDofsChanged_ ()
{
  events_ |= NEW_SLAVE_DOFS_;
}


//-----------------------------------------------------------------------
//   matValuesChanged_
//-----------------------------------------------------------------------


void PointConstrainer::matValuesChanged_ ()
{
  events_ |= NEW_MAT_VALUES_;
}


//-----------------------------------------------------------------------
//   matStructChanged_
//-----------------------------------------------------------------------


void PointConstrainer::matStructChanged_ ()
{
  events_ = NEW_MAT_STRUCT_;
}


//=======================================================================
//   DirectPointConstrainer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DirectPointConstrainer::DirectPointConstrainer

  ( const String&         name,
    Ref<Constraints>      cons,
    Ref<SparseMatrixObj>  matrix ) :

    Super ( name, cons, matrix, matrix )

{
  zappedMatrix_ = newInstance<SparseMatrixObj> ( myName_ );
  warnFlag_     = false;

  print ( System::debug(), myName_,
          " : operating in direct mode;\n"
          "  matrix structure will not be modified.\n" );
}


DirectPointConstrainer::~DirectPointConstrainer ()
{}


//-----------------------------------------------------------------------
//   doUpdate_
//-----------------------------------------------------------------------


void DirectPointConstrainer::doUpdate_ ()
{
  if      ( events_ & NEW_MAT_STRUCT_ )
  {
    updateStructure_ ();
  }
  else if ( events_ & NEW_SLAVE_DOFS_ )
  {
    if ( ! (events_ & NEW_MAT_VALUES_) )
    {
      restoreValues_ ();
    }

    updateStructure_ ();
  }
  else if ( events_ & NEW_MAT_VALUES_ )
  {
    updateValues_ ();
  }
  else if ( events_ & NEW_RVALUES_ )
  {
    updateRhs_    ();
  }
}


//-----------------------------------------------------------------------
//   updateRhs_
//-----------------------------------------------------------------------


void DirectPointConstrainer::updateRhs_ ()
{
  Vector  tmp ( dofMask_.size() );

  tmp              =  0.0;
  tmp[slaveDofs_]  = -rvalues_;

  zappedMatrix_->matmul ( rhs_, tmp );

  rhs_[slaveDofs_] = 0.0;
}


//-----------------------------------------------------------------------
//   updateValues_
//-----------------------------------------------------------------------


void DirectPointConstrainer::updateValues_ ()
{
  Vector  matrixValues = output_      ->getValues ();
  Vector  zappedValues = zappedMatrix_->getValues ();

  print ( System::debug(), myName_,
          " : updating matrix values ...\n" );

  zappedValues = matrixValues[zappedEntries_];

  matrixValues[zappedEntries_]     = 0.0;
  matrixValues[zappedDiagEntries_] = 1.0;

  updateRhs_ ();

  output_      ->setValues ( matrixValues );
  zappedMatrix_->setValues ( zappedValues );
}


//-----------------------------------------------------------------------
//   restoreValues_
//-----------------------------------------------------------------------


void DirectPointConstrainer::restoreValues_ ()
{
  Vector  values = output_->getValues ();

  values[zappedEntries_] = zappedMatrix_->getValues ();
}


//-----------------------------------------------------------------------
//   updateStructure_
//-----------------------------------------------------------------------


void DirectPointConstrainer::updateStructure_ ()
{
  IdxVector    rowOffsets ( output_->getRowOffsets()    );
  IdxVector    colIndices ( output_->getColumnIndices() );

  const idx_t  dofCount      = dofMask_  .size ();
  const idx_t  slaveDofCount = slaveDofs_.size ();

  IdxVector    zappedOffsets ( dofCount + 1 );
  IdxVector    zappedIndices;

  idx_t        irow;
  idx_t        jcol;
  idx_t        j, k, m, n;


  print ( System::debug(), myName_,
          " : updating matrix structure ...\n" );

  zappedDiagEntries_.resize ( slaveDofCount );

  zappedOffsets[0] = 0;

  k = 0;

  for ( irow = 0; irow < dofCount; irow++ )
  {
    m = rowOffsets[irow + 1];

    if ( ! dofMask_[irow] )
    {
      n = m - rowOffsets[irow];

      for ( j = rowOffsets[irow]; j < m; j++ )
      {
        jcol = colIndices[j];

        if ( jcol == irow )
        {
          zappedDiagEntries_[k++] = j;
          break;
        }
      }
    }
    else
    {
      n = 0;

      for ( j = rowOffsets[irow]; j < m; j++ )
      {
        jcol = colIndices[j];

        if ( ! dofMask_[jcol] )
        {
          n++;
        }
      }
    }

    zappedOffsets[irow + 1] = zappedOffsets[irow] + n;
  }

  if ( k < slaveDofCount )
  {
    warnForSingularMatrix_ ();
    zappedDiagEntries_.reshape ( k );
  }

  n = zappedOffsets[dofCount];

  zappedIndices .resize ( n );
  zappedEntries_.resize ( n );

  k = 0;

  for ( irow = 0; irow < dofCount; irow++ )
  {
    m = rowOffsets[irow + 1];

    for ( j = rowOffsets[irow]; j < m; j++ )
    {
      jcol = colIndices[j];

      if ( ! ( dofMask_[irow] && dofMask_[jcol] ) )
      {
        zappedIndices[k]  = jcol;
        zappedEntries_[k] = j;
        k++;
      }
    }
  }

  zappedMatrix_->setMatrix (
    SparseMatrix ( jem::shape ( dofCount, dofCount ),
                   zappedOffsets, zappedIndices )
  );

  updateValues_ ();
}


//-----------------------------------------------------------------------
//   warnForSingularMatrix_
//-----------------------------------------------------------------------


void DirectPointConstrainer::warnForSingularMatrix_ ()
{
  if ( ! warnFlag_ )
  {
    print ( System::warn(), myName_,
            " : constrained matrix is singular;\n"
            "  try `Filter\' mode if this leads to problems.\n" );

    warnFlag_ = true;
  }
}


//=======================================================================
//   class FilterPointConstrainer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


FilterPointConstrainer::FilterPointConstrainer

  ( const String&         name,
    Ref<Constraints>      cons,
    Ref<AbstractMatrix>   input,
    Ref<SparseMatrixObj>  output ) :

    Super ( name, cons, input, output )

{
  if ( ! input->hasExtension<SparseMatrixExt>() )
  {
    throw jem::IllegalArgumentException (
      getContext (),
      "input matrix does not implement the sparse matrix extension"
    );
  }

  print ( System::debug( myName_ ), myName_,
          " : operating in filter mode;\n"
          "  matrix structure will be modified.\n" );
}


FilterPointConstrainer::~FilterPointConstrainer ()
{}


//-----------------------------------------------------------------------
//   doUpdate_
//-----------------------------------------------------------------------


void FilterPointConstrainer::doUpdate_ ()
{
  if      ( events_ & (NEW_MAT_STRUCT_ | NEW_SLAVE_DOFS_) )
  {
    updateStruct_ ();
  }
  else if ( events_ & NEW_MAT_VALUES_ )
  {
    updateValues_ ();
  }
  else if ( events_ & NEW_RVALUES_ )
  {
    updateRhs_    ();
  }
}


//-----------------------------------------------------------------------
//   updateRhs_
//-----------------------------------------------------------------------


void FilterPointConstrainer::updateRhs_ ()
{
  Vector  tmp ( dofMask_.size() );

  tmp              =  0.0;
  tmp[slaveDofs_]  = -rvalues_;

  input_->matmul ( rhs_, tmp );

  rhs_[slaveDofs_] = 0.0;
}


//-----------------------------------------------------------------------
//   updateValues_
//-----------------------------------------------------------------------


void FilterPointConstrainer::updateValues_ ()
{
  SparseMatrixExt*  sx  = input_->getExtension<SparseMatrixExt> ();

  SparseMatrix  input   = sx->toSparseMatrix ();

  IdxVector    inOffsets  = input   .getRowOffsets    ();
  IdxVector    inIndices  = input   .getColumnIndices ();
  Vector       inValues   = input   .getValues        ();
  IdxVector    outOffsets = output_->getRowOffsets    ();
  Vector       outValues  = output_->getValues        ();

  const idx_t  dofCount   = dofMask_.size ();

  idx_t        irow;
  idx_t        jcol;
  idx_t        i, j, n;


  print ( System::debug( myName_ ), myName_,
          " : updating matrix values ...\n" );

  for ( irow = 0; irow < dofCount; irow++ )
  {
    if ( ! dofMask_[irow] )
    {
      outValues[outOffsets[irow]] = 1.0;
      continue;
    }

    n = inOffsets[irow + 1];
    j = outOffsets[irow];

    for ( i = inOffsets[irow]; i < n; i++ )
    {
      jcol = inIndices[i];

      if ( dofMask_[jcol] )
      {
        outValues[j++] = inValues[i];
      }
    }
  }

  updateRhs_ ();

  output_->setValues ( outValues );
}


//-----------------------------------------------------------------------
//   updateStruct_
//-----------------------------------------------------------------------


void FilterPointConstrainer::updateStruct_ ()
{
  using jem::numeric::SparseUtils;

  SparseMatrixExt*  sx  = input_->getExtension<SparseMatrixExt> ();

  SparseMatrix  input   = sx->toSparseMatrix ();

  IdxVector    inOffsets  = input.getRowOffsets    ();
  IdxVector    inIndices  = input.getColumnIndices ();
  Vector       inValues   = input.getValues        ();

  const idx_t  dofCount   = dofMask_.size ();

  IdxVector    outOffsets ( dofCount + 1 );
  IdxVector    outIndices;
  Vector       outValues;

  idx_t        irow;
  idx_t        jcol;
  idx_t        i, j, n;


  print ( System::debug( myName_ ), myName_,
          " : updating matrix structure ...\n" );

  output_->setMatrix ( SparseMatrix() );

  outOffsets = 0_idx;

  for ( irow = 0; irow < dofCount; irow++ )
  {
    if ( ! dofMask_[irow] )
    {
      outOffsets[irow] = 1;
      continue;
    }

    n = inOffsets[irow + 1];

    for ( i = inOffsets[irow]; i < n; i++ )
    {
      jcol = inIndices[i];

      if ( dofMask_[jcol] )
      {
        outOffsets[irow]++;
      }
    }
  }

  SparseUtils::sumOffsets ( outOffsets );

  outIndices.resize ( outOffsets.back() );
  outValues .resize ( outOffsets.back() );

  for ( irow = 0; irow < dofCount; irow++ )
  {
    if ( ! dofMask_[irow] )
    {
      j = outOffsets[irow];

      outIndices[j] = irow;
      outValues [j] = 1.0;

      continue;
    }

    n = inOffsets[irow + 1];
    j = outOffsets[irow];

    for ( i = inOffsets[irow]; i < n; i++ )
    {
      jcol = inIndices[i];

      if ( dofMask_[jcol] )
      {
        outIndices[j] = jcol;
        outValues [j] = inValues[i];
        j++;
      }
    }
  }

  updateRhs_ ();

  output_->setMatrix (
    SparseMatrix ( jem::shape ( dofCount, dofCount ),
                   outOffsets, outIndices, outValues )
  );
}


JIVE_END_PACKAGE( solver )
