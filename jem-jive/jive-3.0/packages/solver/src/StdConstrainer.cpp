
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
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/MemoryError.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Event.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/utilities.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/solver/Names.h>
#include <jive/solver/ConstrainerFactory.h>
#include <jive/solver/StdConstrainer.h>


JEM_DEFINE_CLASS( jive::solver::StdConstrainer );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::System;
using jive::algebra::SparseMatrixExt;

typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;


//=======================================================================
//   class StdConstrainer::Data_
//=======================================================================


class StdConstrainer::Data_ : public Collectable
{
 public:

  typedef Data_           Self;
  typedef Collectable     Super;

  static const int        NEW_RVALUES    = 1 << 0;
  static const int        NEW_CON_STRUCT = 1 << 1;
  static const int        NEW_MAT_VALUES = 1 << 2;
  static const int        NEW_MAT_STRUCT = 1 << 3;;


                          Data_

    ( Ref<Constraints>      cons,
      Ref<AbstractMatrix>   input,
      Ref<SparseMatrixObj>  output );


 public:

  Ref<Constraints>        cons;
  Ref<AbstractMatrix>     input;
  SparseMatrixExt*        xinput;
  Ref<SparseMatrixObj>    output;

  SparseMatrix            helpMat;
  SparseMatrix            conMat;
  IdxVector               conMap;
  Vector                  rhs;

  bool                    distributed;
  int                     events;


 protected:

  virtual                ~Data_               ();


 private:

  void                    rvaluesChanged_     ();
  void                    conStructChanged_   ();
  void                    matValuesChanged_   ();
  void                    matStructChanged_   ();

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdConstrainer::Data_::Data_

  ( Ref<Constraints>      cs,
    Ref<AbstractMatrix>   in,
    Ref<SparseMatrixObj>  out ) :

    cons   (  cs ),
    input  (  in ),
    output ( out )

{
  using jem::util::connect;

  xinput       = input->getExtension<SparseMatrixExt> ();
  distributed  = in->isDistributed ();
  events       = ~0x0;

  connect ( cons->newRvaluesEvent,
            this, & Self::rvaluesChanged_ );

  connect ( cons->newStructEvent,
            this, & Self::conStructChanged_ );

  connect ( input->newValuesEvent,
            this, & Self::matValuesChanged_ );

  connect ( input->newStructEvent,
            this, & Self::matStructChanged_ );

  // This is to make sure that the output matrix is updated
  // if some moron has modified it.

  connect ( output->newValuesEvent,
            this, & Self::matValuesChanged_ );

  connect ( output->newStructEvent,
            this, & Self::matStructChanged_ );
}


StdConstrainer::Data_::~Data_ ()
{}


//-----------------------------------------------------------------------
//   rvaluesChanged_
//-----------------------------------------------------------------------


void StdConstrainer::Data_::rvaluesChanged_ ()
{
  events |= NEW_RVALUES;
}


//-----------------------------------------------------------------------
//   conStructChanged_
//-----------------------------------------------------------------------


void StdConstrainer::Data_::conStructChanged_ ()
{
  events |= NEW_CON_STRUCT;
}


//-----------------------------------------------------------------------
//   matValuesChanged_
//-----------------------------------------------------------------------


void StdConstrainer::Data_::matValuesChanged_ ()
{
  events |= NEW_MAT_VALUES;
}


//-----------------------------------------------------------------------
//   matStructChanged_
//-----------------------------------------------------------------------


void StdConstrainer::Data_::matStructChanged_ ()
{
  events |= NEW_MAT_STRUCT;
}


//=======================================================================
//   class StdConstrainer::Utils_
//=======================================================================


class StdConstrainer::Utils_
{
 public:

  static IdxVector            makeConMap

    ( const SparseMatrix&       conMat );

  static SparseMatrix         makeHelpMatrix

    ( const SparseMatrix&       inMat,
      const SparseMatrix&       conMat,
      const IdxVector&          conMap );

  static SparseMatrix         makeOutputMatrix

    ( const SparseMatrix&       helpMat,
      const SparseMatrix&       inMat,
      const SparseMatrix&       conMat,
      const IdxVector&          conMap );

  static void                 updateHelpMatrix

    ( const SparseMatrix&       helpMat,
      const SparseMatrix&       inMat,
      const SparseMatrix&       conMat,
      const IdxVector&          conMap );

  static void                 updateOutputMatrix

    ( const SparseMatrix&       outMat,
      const SparseMatrix&       helpMat,
      const SparseMatrix&       inMat,
      const SparseMatrix&       conMat,
      const IdxVector&          conMap );

  static void                 getRowStructure

    ( IdxBuffer&                rowBuf,
      const BoolVector&         mask,
      const IdxVector&          conMap,
      const IdxVector&          inIndices,
      const IdxVector&          conOffsets,
      const IdxVector&          conIndices,
      idx_t                     first,
      idx_t                     last );

  static void                 addRowStructure

    ( IdxBuffer&                rowBuf,
      const BoolVector&         mask,
      const IdxVector&          indices,
      idx_t                     first,
      idx_t                     last );

  static void                 getRowValues

    ( double* JEM_RESTRICT      accu,
      const IdxVector&          conMap,
      const IdxVector&          inIndices,
      const Vector&             inValues,
      const IdxVector&          conOffsets,
      const IdxVector&          conIndices,
      const Vector&             conValues,
      idx_t                     first,
      idx_t                     last );

  static inline void          addRowValues

    ( double* JEM_RESTRICT      accu,
      const IdxVector&          indices,
      const Vector&             values,
      double                    scale,
      idx_t                     first,
      idx_t                     last );

};


//-----------------------------------------------------------------------
//   makeConMap
//-----------------------------------------------------------------------


IdxVector StdConstrainer::Utils_::makeConMap

  ( const SparseMatrix& conMat )

{
  const idx_t  conCount   = conMat.size(0);
  const idx_t  dofCount   = conMat.size(1);

  IdxVector    conOffsets = conMat.getRowOffsets    ();
  IdxVector    conIndices = conMat.getColumnIndices ();

  IdxVector    conMap     ( dofCount );


  conMap = -1;

  for ( idx_t icon = 0; icon < conCount; icon++ )
  {
    conMap[conIndices[conOffsets[icon]]] = icon;
  }

  return conMap;
}


//-----------------------------------------------------------------------
//   makeHelpMatrix
//-----------------------------------------------------------------------


SparseMatrix StdConstrainer::Utils_::makeHelpMatrix

  ( const SparseMatrix&  inMat,
    const SparseMatrix&  conMat,
    const IdxVector&     conMap )

{
  const idx_t  conCount    = conMat.size(0);
  const idx_t  dofCount    = conMat.size(1);

  IdxVector    inOffsets   = inMat .getRowOffsets    ();
  IdxVector    inIndices   = inMat .getColumnIndices ();
  IdxVector    conOffsets  = conMat.getRowOffsets    ();
  IdxVector    conIndices  = conMat.getColumnIndices ();

  BoolVector   mask        ( dofCount );
  IdxVector    helpOffsets ( conCount + 1 );
  IdxBuffer    helpIndices;


  mask = true;

  helpIndices.reserve ( conIndices.size() );

  for ( idx_t icon = 0; icon < conCount; icon++ )
  {
    helpOffsets[icon] = helpIndices.size ();

    idx_t  irow       = conIndices[conOffsets[icon]];

    getRowStructure ( helpIndices,
                      mask,
                      conMap,
                      inIndices,
                      conOffsets,
                      conIndices,
                      inOffsets[irow],
                      inOffsets[irow + 1] );

    idx_t  n = helpIndices.size ();

    for ( idx_t i = helpOffsets[icon]; i < n; i++ )
    {
      mask[helpIndices[i]] = true;
    }
  }

  helpOffsets[conCount] = helpIndices.size ();

  helpIndices.trimToSize ();

  return SparseMatrix ( jem::shape ( conCount, dofCount ),
                        helpOffsets,
                        helpIndices.toArray() );
}


//-----------------------------------------------------------------------
//   makeOutputMatrix
//-----------------------------------------------------------------------


SparseMatrix StdConstrainer::Utils_::makeOutputMatrix

  ( const SparseMatrix&  helpMat,
    const SparseMatrix&  inMat,
    const SparseMatrix&  conMat,
    const IdxVector&     conMap )

{
  SparseMatrix  tconMat     = conMat.transpose ();

  IdxVector     tconOffsets = tconMat.getRowOffsets    ();
  IdxVector     tconIndices = tconMat.getColumnIndices ();
  IdxVector     helpOffsets = helpMat.getRowOffsets    ();
  IdxVector     helpIndices = helpMat.getColumnIndices ();
  IdxVector     inOffsets   = inMat  .getRowOffsets    ();
  IdxVector     inIndices   = inMat  .getColumnIndices ();
  IdxVector     conOffsets  = conMat .getRowOffsets    ();
  IdxVector     conIndices  = conMat .getColumnIndices ();

  const idx_t   dofCount    = conMat.size(1);

  BoolVector    mask        ( dofCount );
  IdxVector     outOffsets  ( dofCount + 1 );
  IdxBuffer     outIndices;


  mask = true;

  outIndices.reserve ( inIndices.size() );

  for ( idx_t irow = 0; irow < dofCount; irow++ )
  {
    outOffsets[irow] = outIndices.size ();

    if ( conMap[irow] >= 0 )
    {
      outIndices.pushBack ( irow );
      continue;
    }

    getRowStructure ( outIndices,
                      mask,
                      conMap,
                      inIndices,
                      conOffsets,
                      conIndices,
                      inOffsets[irow],
                      inOffsets[irow + 1] );

    idx_t  n = tconOffsets[irow + 1];

    for ( idx_t i = tconOffsets[irow]; i < n; i++ )
    {
      idx_t  icon = tconIndices[i];

      addRowStructure ( outIndices,
                        mask,
                        helpIndices,
                        helpOffsets[icon],
                        helpOffsets[icon + 1] );
    }

    n = outIndices.size ();

    for ( idx_t i = outOffsets[irow]; i < n; i++ )
    {
      mask[outIndices[i]] = true;
    }
  }

  outOffsets[dofCount] = outIndices.size ();

  outIndices.trimToSize ();

  return SparseMatrix ( jem::shape ( dofCount, dofCount ),
                        outOffsets,
                        outIndices.toArray() );
}


//-----------------------------------------------------------------------
//   updateHelpMatrix
//-----------------------------------------------------------------------


void StdConstrainer::Utils_::updateHelpMatrix

  ( const SparseMatrix&  helpMat,
    const SparseMatrix&  inMat,
    const SparseMatrix&  conMat,
    const IdxVector&     conMap )

{
  const idx_t  conCount    = conMat.size(0);
  const idx_t  dofCount    = conMat.size(1);

  IdxVector    helpOffsets = helpMat.getRowOffsets    ();
  IdxVector    helpIndices = helpMat.getColumnIndices ();
  Vector       helpValues  = helpMat.getValues        ();
  IdxVector    inOffsets   = inMat  .getRowOffsets    ();
  IdxVector    conOffsets  = conMat .getRowOffsets    ();
  IdxVector    conIndices  = conMat .getColumnIndices ();

  Vector       accu ( dofCount + 1 );


  accu = 0.0;

  for ( idx_t icon = 0; icon < conCount; icon++ )
  {
    idx_t  irow = conIndices[conOffsets[icon]];

    getRowValues ( accu .addr (),
                   conMap,
                   inMat.getColumnIndices (),
                   inMat.getValues        (),
                   conOffsets,
                   conIndices,
                   conMat.getValues (),
                   inOffsets[irow],
                   inOffsets[irow + 1] );

    idx_t  n = helpOffsets[icon + 1];

    for ( idx_t i = helpOffsets[icon]; i < n; i++ )
    {
      idx_t  jcol   = helpIndices[i];

      helpValues[i] = accu[jcol];
      accu[jcol]    = 0.0;
    }
  }

  if ( ! jem::isTiny( accu[dofCount] ) )
  {
    throw jem::MemoryError ( JEM_FUNC );
  }
}


//-----------------------------------------------------------------------
//   updateOutputMatrix
//-----------------------------------------------------------------------


void StdConstrainer::Utils_::updateOutputMatrix

  ( const SparseMatrix&  outMat,
    const SparseMatrix&  helpMat,
    const SparseMatrix&  inMat,
    const SparseMatrix&  conMat,
    const IdxVector&     conMap )

{
  SparseMatrix  tconMat     = conMat.transpose ();

  IdxVector     tconOffsets = tconMat.getRowOffsets    ();
  IdxVector     tconIndices = tconMat.getColumnIndices ();
  Vector        tconValues  = tconMat.getValues        ();
  IdxVector     outOffsets  = outMat .getRowOffsets    ();
  IdxVector     outIndices  = outMat .getColumnIndices ();
  Vector        outValues   = outMat .getValues        ();
  IdxVector     helpOffsets = helpMat.getRowOffsets    ();
  IdxVector     inOffsets   = inMat  .getRowOffsets    ();

  const idx_t   dofCount    = conMat.size(1);

  Vector        accu        ( dofCount + 1 );


  accu = 0.0;

  for ( idx_t irow = 0; irow < dofCount; irow++ )
  {
    if ( conMap[irow] >= 0 )
    {
      outValues[outOffsets[irow]] = 1.0;
      continue;
    }

    getRowValues ( accu  .addr (),
                   conMap,
                   inMat .getColumnIndices (),
                   inMat .getValues        (),
                   conMat.getRowOffsets    (),
                   conMat.getColumnIndices (),
                   conMat.getValues        (),
                   inOffsets[irow],
                   inOffsets[irow + 1] );

    idx_t  n = tconOffsets[irow + 1];

    for ( idx_t i = tconOffsets[irow]; i < n; i++ )
    {
      idx_t  icon = tconIndices[i];

      addRowValues ( accu   .addr (),
                     helpMat.getColumnIndices (),
                     helpMat.getValues        (),
                     tconValues[i],
                     helpOffsets[icon],
                     helpOffsets[icon + 1] );
    }

    n = outOffsets[irow + 1];

    for ( idx_t i = outOffsets[irow]; i < n; i++ )
    {
      idx_t  jcol  = outIndices[i];

      outValues[i] = accu[jcol];
      accu[jcol]   = 0.0;
    }
  }

  if ( ! jem::isTiny( accu[dofCount] ) )
  {
    throw jem::MemoryError ( JEM_FUNC );
  }
}


//-----------------------------------------------------------------------
//   getRowStructure
//-----------------------------------------------------------------------


void StdConstrainer::Utils_::getRowStructure

  ( IdxBuffer&         rowBuf,
    const BoolVector&  mask,
    const IdxVector&   conMap,
    const IdxVector&   inIndices,
    const IdxVector&   conOffsets,
    const IdxVector&   conIndices,
    idx_t              first,
    idx_t              last )

{
  for ( idx_t i = first; i < last; i++ )
  {
    idx_t  jcol = inIndices[i];

    if ( conMap[jcol] < 0 )
    {
      if ( mask[jcol] )
      {
        rowBuf.pushBack ( jcol );
        mask[jcol] = false;
      }
    }
    else
    {
      idx_t  icon = conMap[jcol];
      idx_t  n    = conOffsets[icon + 1];

      for ( idx_t j = conOffsets[icon] + 1; j < n; j++ )
      {
        jcol = conIndices[j];

        if ( mask[jcol] )
        {
          rowBuf.pushBack ( jcol );
          mask[jcol] = false;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   addRowStructure
//-----------------------------------------------------------------------


void StdConstrainer::Utils_::addRowStructure

  ( IdxBuffer&         rowBuf,
    const BoolVector&  mask,
    const IdxVector&   indices,
    idx_t              first,
    idx_t              last )

{
  for ( idx_t i = first; i < last; i++ )
  {
    idx_t  jcol = indices[i];

    if ( mask[jcol] )
    {
      rowBuf.pushBack ( jcol );
      mask[jcol] = false;
    }
  }
}


//-----------------------------------------------------------------------
//   getRowValues
//-----------------------------------------------------------------------


void StdConstrainer::Utils_::getRowValues

  ( double* JEM_RESTRICT  accu,
    const IdxVector&      conMap,
    const IdxVector&      inIndices,
    const Vector&         inValues,
    const IdxVector&      conOffsets,
    const IdxVector&      conIndices,
    const Vector&         conValues,
    idx_t                 first,
    idx_t                 last )

{
  for ( idx_t i = first; i < last; i++ )
  {
    double  val  = inValues[i];
    idx_t   jcol = inIndices[i];

    if ( conMap[jcol] < 0 )
    {
      accu[jcol] += val;
    }
    else
    {
      idx_t  icon = conMap[jcol];
      idx_t  j    = conOffsets[icon];
      idx_t  n    = conOffsets[icon + 1];

JEM_IVDEP

      for ( j++; j < n; j++ )
      {
        accu[conIndices[j]] += val * conValues[j];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   addRowValues
//-----------------------------------------------------------------------


inline void StdConstrainer::Utils_::addRowValues

  ( double* JEM_RESTRICT  accu,
    const IdxVector&      indices,
    const Vector&         values,
    double                scale,
    idx_t                 first,
    idx_t                 last )

{
JEM_IVDEP

  for ( idx_t i = first; i < last; i++ )
  {
    accu[indices[i]] += scale * values[i];
  }
}


//=======================================================================
//   class StdConstrainer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  StdConstrainer::TYPE_NAME = "Standard";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdConstrainer::StdConstrainer

  ( const String&         name,
    Ref<Constraints>      cons,
    Ref<AbstractMatrix>   input,
    Ref<SparseMatrixObj>  output ) :

    Super ( name )

{
  using jive::util::joinNames;

  JEM_PRECHECK ( cons && input && input != output );

  if ( ! input->hasExtension<SparseMatrixExt>() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      "input matrix does not implement the sparse matrix extension"
    );
  }

  if ( input->isDistributed() )
  {
    print ( System::warn(), myName_,
            " : distributed matrix;\n"
            "  constrained matrix may be incorrect\n" );
  }

  if ( ! output )
  {
    SparseMatrixObj::Traits  traits = 0;

    if ( input->isSymmetric() )
    {
      traits |= SparseMatrixObj::SYMMETRIC;
    }

    output = newInstance<SparseMatrixObj>

      ( joinNames( myName_, PropNames::MATRIX ), traits );
  }

  data_ = newInstance<Data_> ( cons, input, output );
}


StdConstrainer::~StdConstrainer ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void StdConstrainer::update ()
{
  Data_&  d        = * data_;

  idx_t   dofCount = d.conMat.size(1);
  idx_t   n        = d.input->shape()[0];


  if ( n != dofCount )
  {
    if ( ! d.input->isSquare() )
    {
      util::nonSquareMatrixError ( getContext (),
                                   d.input->shape() );
    }

    dofCount = n;

    d.rhs.resize ( dofCount );

    d.events |= (Data_::NEW_CON_STRUCT | Data_::NEW_MAT_STRUCT);
  }

  if ( d.events & Data_::NEW_CON_STRUCT )
  {
    d.conMat = SparseMatrix ();
    d.conMat = d.cons->toMatrix ();

    if ( d.conMat.size(1) != dofCount )
    {
      throw jem::IllegalArgumentException (
        JEM_FUNC,
        "incompatible matrix and constraints "
        "(unequal matrix size and number of dofs)"
      );
    }

    d.conMap.ref ( Utils_::makeConMap( d.conMat ) );
  }
  else if ( d.events & Data_::NEW_RVALUES )
  {
    d.cons->getRvalues ( d.conMat );
  }

  // Take a shortcut if no constraints have been defined

  if ( d.conMat.size(0) == 0 && ! d.distributed )
  {
    if      ( d.events & (Data_::NEW_MAT_STRUCT |
                          Data_::NEW_CON_STRUCT) )
    {
      d.output->setMatrix ( d.xinput->toSparseMatrix() );
    }
    else if ( d.events & Data_::NEW_MAT_VALUES )
    {
      SparseMatrix  sm = d.xinput->toSparseMatrix ();

      d.output->setValues ( sm.getValues() );
    }

    if ( d.events )
    {
      d.rhs = 0.0;
    }
  }
  else
  {
    if ( d.events & (Data_::NEW_RVALUES | Data_::NEW_CON_STRUCT) )
    {
      double  chksum = jem::sum ( d.conMat.getValues() );

      if ( jem::Float::isNaN( chksum ) )
      {
        throw jem::ArithmeticException (
          getContext (),
          "invalid constraints (NaN)"
        );
      }
    }

    if      ( d.events & (Data_::NEW_MAT_STRUCT |
                          Data_::NEW_CON_STRUCT) )
    {
      updateStruct_ ();
    }
    else if ( d.events & Data_::NEW_MAT_VALUES )
    {
      updateValues_ ();
    }
    else if ( d.events & Data_::NEW_RVALUES )
    {
      updateRhs_    ();
    }
  }

  d.cons  ->resetEvents ();
  d.input ->resetEvents ();
  d.output->resetEvents ();

  d.events = 0;
}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


bool StdConstrainer::updated () const
{
  return (data_->events == 0);
}


//-----------------------------------------------------------------------
//   getRhs
//-----------------------------------------------------------------------


void StdConstrainer::getRhs

  ( const Vector&  g,
    const Vector&  f ) const

{
  JEM_PRECHECK ( updated() && g.size() == f.size() );

  const Data_&  d = * data_;

  const idx_t   dofCount = d.conMat.size (1);


  if ( f.size() != dofCount )
  {
    util::sizeError ( getContext(), "rhs vector", f.size(), dofCount );
  }

  g = f - d.rhs;

  util::evalMasterDofs ( g, *d.cons );
}


//-----------------------------------------------------------------------
//   getLhs
//-----------------------------------------------------------------------


void StdConstrainer::getLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  JEM_PRECHECK ( updated() && x.size() == y.size() );

  const Data_&  d = * data_;

  const idx_t   conCount = d.conMat.size (0);
  const idx_t   dofCount = d.conMat.size (1);


  // Take a shortcut if no constraints have been defined

  if ( conCount == 0 )
  {
    x = y;

    return;
  }

  if ( x.size() != dofCount )
  {
    util::sizeError ( getContext(), "lhs vector", x.size(), dofCount );
  }

  x = y;

  util::evalSlaveDofs ( x, *d.cons );
}


//-----------------------------------------------------------------------
//   initLhs
//-----------------------------------------------------------------------


void StdConstrainer::initLhs

  ( const Vector&  x,
    const Vector&  y ) const

{
  JEM_PRECHECK ( updated() && x.size() == y.size() );

  const Data_&  d = * data_;

  const idx_t   conCount = d.conMat.size (0);
  const idx_t   dofCount = d.conMat.size (1);


  // Take a shortcut if no constraints have been defined

  if ( conCount == 0 )
  {
    x = y;

    return;
  }

  if ( x.size() != dofCount )
  {
    util::sizeError ( getContext(), "lhs vector", x.size(), dofCount );
  }

  x = y;

  util::zeroSlaveDofs ( x, *d.cons );
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* StdConstrainer::getConstraints () const
{
  return data_->cons.get ();
}


//-----------------------------------------------------------------------
//   getInputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* StdConstrainer::getInputMatrix () const
{
  return data_->input.get ();
}


//-----------------------------------------------------------------------
//   getOutputMatrix
//-----------------------------------------------------------------------


AbstractMatrix* StdConstrainer::getOutputMatrix () const
{
  return data_->output.get ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Constrainer> StdConstrainer::makeNew

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix )

{
  if ( matrix->hasExtension<SparseMatrixExt>() )
  {
    return newInstance<Self> ( name, cons, matrix );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdConstrainer::declare ()
{
  ConstrainerFactory::declare ( TYPE_NAME,  & makeNew );
  ConstrainerFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   updateRhs_
//-----------------------------------------------------------------------


void StdConstrainer::updateRhs_ ()
{
  const Data_&  d = * data_;

  const idx_t  dofCount = d.conMat.size (1);

  Vector       tmp ( dofCount );


  JEM_PRECHECK ( d.rhs.size() == dofCount );

  tmp = 0.0;

  util::setSlaveDofs ( tmp,   *d.cons );
  d.input->matmul    ( d.rhs,  tmp    );
}


//-----------------------------------------------------------------------
//   updateValues_
//-----------------------------------------------------------------------


void StdConstrainer::updateValues_ ()
{
  const Data_&  d      = * data_;

  SparseMatrix  inMat  = d.xinput->toSparseMatrix ();
  SparseMatrix  outMat = d.output->toSparseMatrix ();

  JEM_PRECHECK ( d.input != d.output );


  print ( System::debug( myName_ ), myName_,
          " : updating matrix values ...\n" );

  Utils_::updateHelpMatrix   ( d.helpMat,
                               inMat,
                               d.conMat,
                               d.conMap );

  Utils_::updateOutputMatrix ( outMat,
                               d.helpMat,
                               inMat,
                               d.conMat,
                               d.conMap );

  updateRhs_ ();

  d.output->setValues ( outMat.getValues() );
}


//-----------------------------------------------------------------------
//   updateStruct_
//-----------------------------------------------------------------------


void StdConstrainer::updateStruct_ ()
{
  Data_&        d     = * data_;

  SparseMatrix  inMat = d.xinput->toSparseMatrix ();

  SparseMatrix  outMat;
  SparseMatrix  emptyMat;


  print ( System::debug( myName_ ), myName_,
          " : updating matrix structure ...\n" );

  d.output->setMatrix ( emptyMat );

  d.helpMat = emptyMat;

  d.helpMat = Utils_::makeHelpMatrix   ( inMat,
                                         d.conMat,
                                         d.conMap );

  outMat    = Utils_::makeOutputMatrix ( d.helpMat,
                                         inMat,
                                         d.conMat,
                                         d.conMap );

  Utils_::updateHelpMatrix             ( d.helpMat,
                                         inMat,
                                         d.conMat,
                                         d.conMap );

  Utils_::updateOutputMatrix           ( outMat,
                                         d.helpMat,
                                         inMat,
                                         d.conMat,
                                         d.conMap );

  updateRhs_ ();

  d.output->setMatrix ( outMat );
}


JIVE_END_PACKAGE( solver )
