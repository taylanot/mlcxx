
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

/*
  The class SkylineSolver uses Crout's algorithm to decompose a sparse
  matrix A into a lower triangular factor L and an upper triangular
  factor U:

    A = L * U

  Both the matrix and the factors are viewed as band matrices with a
  variable bandwith. The bandwidth is minimized by permuting the rows
  and columns of the matrix before the factors are computed. The
  precision of the solver is improved by scaling each matrix row with
  the largest absolute entry in that row.

  The entries in the upper factor are stored column wise and the
  entries in the lower factor are stored row wise. To increase the
  performance of the solver, each group of four consecutive columns in
  the upper factor is stored as a dense, rectangular (n x 4) matrix,
  with n the height of the column rounded up to the nearest multiple
  of four. These matrices are called the "upper blocks". The entries
  in the upper blocks are stored row wise. Likewise, each group of
  four consecutive rows in the lower factor is stored as a dense (4 x
  n) matrix with n the width of the row rounded up to nearest multiple
  of four. These matrices are called the "lower blocks"; their entries
  are stored column wise.  column wise.

  The entries near the diagonal of the matrix are stored row wise in
  dense (4 x 4) matrices called "pivot blocks". The upper triangle of
  a pivot block is part of the upper factor, while the lower triangle
  is part of the lower factor. The diagonal entries of the pivot
  blocks are the diagonal entries of the upper factor.
*/

#include <cmath>
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/Thread.h>
#include <jem/base/Monitor.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/CancelledException.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/Reorder.h>
#include <jem/util/Flex.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverInfo.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/SolverFactory.h>
#include <jive/solver/SolverException.h>
#include <jive/solver/StdConstrainer.h>
#include <jive/solver/DummyConstrainer.h>
#include <jive/solver/SkylineSolver.h>


JEM_DEFINE_CLASS( jive::solver::SkylineSolver );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::Lock;
using jem::Array;
using jem::Float;
using jem::System;
using jem::Thread;
using jem::Monitor;
using jem::CancelledException;
using jem::ArithmeticException;
using jem::IllegalInputException;
using jem::io::endl;
using jem::util::Flex;
using jive::algebra::SparseMatrixExt;


//=======================================================================
//   class SkylineSolver::Data_
//=======================================================================


class SkylineSolver::Data_ : public jem::Collectable
{
 public:

  inline                    Data_         ();

  void                      resize

    ( idx_t                   msize );

  void                      reorder

    ( const SparseStruct&     mstruct );

  inline idx_t              matrixSize    () const;
  inline idx_t              blockCount    () const;
  double                    getMemUsage   () const;

  double                    factor4Rows

    ( idx_t                   iblk );

  double                    factor4Columns

    ( idx_t                   jblk );


 public:

  IdxVector                 iperm;
  IdxVector                 invPerm;
  Vector                    rowScale;
  IdxVector                 startRows;
  IdxVector                 startCols;
  Flex<idx_t>               zeroPivots;
  Vector                    tempVector;
  Vector                    pivotBlocks;
  Flex<double>              lowerBlocks;
  Flex<double>              upperBlocks;
  IdxVector                 lowerOffsets;
  IdxVector                 upperOffsets;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline SkylineSolver::Data_::Data_ ()
{}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


void SkylineSolver::Data_::resize ( idx_t msize )
{
  const idx_t  bcount = (msize + 3) / 4;

  iperm       .resize ( msize );
  invPerm     .resize ( msize );
  rowScale    .resize ( msize );
  startRows   .resize ( bcount );
  startCols   .resize ( bcount );
  zeroPivots  .resize ( 0 );
  tempVector  .resize ( bcount * 4 );
  pivotBlocks .resize ( bcount * 4 * 4 );
  lowerBlocks .resize ( 0 );
  upperBlocks .resize ( 0 );
  lowerOffsets.resize ( bcount + 1 );
  upperOffsets.resize ( bcount + 1 );
}


//-----------------------------------------------------------------------
//   reorder
//-----------------------------------------------------------------------


void SkylineSolver::Data_::reorder ( const SparseStruct& mstruct )
{
  using jem::numeric::Reorder;

  JEM_ASSERT ( iperm.size() == mstruct.size(0) );

  const idx_t  msize = iperm.size ();

  Reorder::superReorder ( invPerm, mstruct, & Reorder::rcm );

  // Invert the permutation array.

  for ( idx_t i = 0; i < msize; i++ )
  {
    iperm[invPerm[i]] = i;
  }
}


//-----------------------------------------------------------------------
//   matrixSize
//-----------------------------------------------------------------------


inline idx_t SkylineSolver::Data_::matrixSize () const
{
  return iperm.size ();
}


//-----------------------------------------------------------------------
//   blockCount
//-----------------------------------------------------------------------


inline idx_t SkylineSolver::Data_::blockCount () const
{
  return startRows.size ();
}


//-----------------------------------------------------------------------
//   getMemUsage
//-----------------------------------------------------------------------


double SkylineSolver::Data_::getMemUsage () const
{
  double  isize  = (double) sizeof(idx_t);
  double  dsize  = (double) sizeof(double);

  double  musage = 0.0;

  musage += isize * (double) iperm       .size ();
  musage += isize * (double) invPerm     .size ();
  musage += dsize * (double) rowScale    .size ();
  musage += isize * (double) startRows   .size ();
  musage += isize * (double) startCols   .size ();
  musage += isize * (double) zeroPivots  .size ();
  musage += dsize * (double) tempVector  .size ();
  musage += dsize * (double) pivotBlocks .size ();
  musage += dsize * (double) lowerBlocks .size ();
  musage += dsize * (double) upperBlocks .size ();
  musage += isize * (double) lowerOffsets.size ();
  musage += isize * (double) upperOffsets.size ();

  return musage;
}


//-----------------------------------------------------------------------
//   factor4Rows
//-----------------------------------------------------------------------


double SkylineSolver::Data_::factor4Rows ( idx_t iblk )
{
  JEM_ASSERT ( iblk >= 0 && iblk < blockCount() );

  const double*  JEM_RESTRICT  piv;
  const double*  JEM_RESTRICT  col;
  double*        JEM_RESTRICT  row;

  double  flops;

  double  a00, a01, a02, a03;
  double  a10, a11, a12, a13;
  double  a20, a21, a22, a23;
  double  a30, a31, a32, a33;

  idx_t   iblk0;
  idx_t   jblk0;
  idx_t   n;


  // Compute the next four rows of the lower factor.

  flops = 0.0;
  jblk0 = startCols[iblk];

  for ( idx_t jblk = jblk0; jblk < iblk; jblk++ )
  {
    iblk0 = startRows[jblk];

    piv   = pivotBlocks.addr ( 16 * jblk );
    row   = lowerBlocks.addr ( lowerOffsets[iblk] );
    col   = upperBlocks.addr ( upperOffsets[jblk] );

    if ( iblk0 < jblk0 )
    {
      n    =  4 * (jblk  - jblk0);
      col += 16 * (jblk0 - iblk0);
    }
    else
    {
      n    =  4 * (jblk  - iblk0);
      row += 16 * (iblk0 - jblk0);
    }

    a00 = a01 = a02 = a03 = 0.0;
    a10 = a11 = a12 = a13 = 0.0;
    a20 = a21 = a22 = a23 = 0.0;
    a30 = a31 = a32 = a33 = 0.0;

    for ( idx_t i = 0; i < n; i++ )
    {
      a00 += row[0] * col[0];
      a10 += row[1] * col[0];
      a20 += row[2] * col[0];
      a30 += row[3] * col[0];

      a01 += row[0] * col[1];
      a11 += row[1] * col[1];
      a21 += row[2] * col[1];
      a31 += row[3] * col[1];

      a02 += row[0] * col[2];
      a12 += row[1] * col[2];
      a22 += row[2] * col[2];
      a32 += row[3] * col[2];

      a03 += row[0] * col[3];
      a13 += row[1] * col[3];
      a23 += row[2] * col[3];
      a33 += row[3] * col[3];

      row += 4;
      col += 4;
    }

    row[0]  = piv[0] * (row[0] - a00);
    row[1]  = piv[0] * (row[1] - a10);
    row[2]  = piv[0] * (row[2] - a20);
    row[3]  = piv[0] * (row[3] - a30);

    a01    += row[0] * piv[1];
    a11    += row[1] * piv[1];
    a21    += row[2] * piv[1];
    a31    += row[3] * piv[1];

    a02    += row[0] * piv[2];
    a12    += row[1] * piv[2];
    a22    += row[2] * piv[2];
    a32    += row[3] * piv[2];

    a03    += row[0] * piv[3];
    a13    += row[1] * piv[3];
    a23    += row[2] * piv[3];
    a33    += row[3] * piv[3];

    row[4]  = piv[5] * (row[4] - a01);
    row[5]  = piv[5] * (row[5] - a11);
    row[6]  = piv[5] * (row[6] - a21);
    row[7]  = piv[5] * (row[7] - a31);

    a02    += row[4] * piv[6];
    a12    += row[5] * piv[6];
    a22    += row[6] * piv[6];
    a32    += row[7] * piv[6];

    a03    += row[4] * piv[7];
    a13    += row[5] * piv[7];
    a23    += row[6] * piv[7];
    a33    += row[7] * piv[7];

    row[8]  = piv[10] * (row[8]  - a02);
    row[9]  = piv[10] * (row[9]  - a12);
    row[10] = piv[10] * (row[10] - a22);
    row[11] = piv[10] * (row[11] - a32);

    a03    += row[8]  * piv[11];
    a13    += row[9]  * piv[11];
    a23    += row[10] * piv[11];
    a33    += row[11] * piv[11];

    row[12] = piv[15] * (row[12] - a03);
    row[13] = piv[15] * (row[13] - a13);
    row[14] = piv[15] * (row[14] - a23);
    row[15] = piv[15] * (row[15] - a33);

    flops  += (40.0 + 16.0 * (double) n);
  }

  return flops;
}


//-----------------------------------------------------------------------
//   factor4Columns
//-----------------------------------------------------------------------


double SkylineSolver::Data_::factor4Columns ( idx_t jblk )
{
  JEM_ASSERT ( jblk >= 0 && jblk < blockCount() );

  const double* JEM_RESTRICT  piv;
  const double* JEM_RESTRICT  row;
  double*       JEM_RESTRICT  col;

  double  flops;

  double  a00, a01, a02, a03;
  double  a10, a11, a12, a13;
  double  a20, a21, a22, a23;
  double  a30, a31, a32, a33;

  idx_t   iblk0;
  idx_t   jblk0;
  idx_t   n;


  // Factor the next four columns of the upper factor.

  flops = 0.0;
  iblk0 = startRows[jblk];

  for ( idx_t iblk = iblk0; iblk < jblk; iblk++ )
  {
    jblk0 = startCols[iblk];

    piv   = pivotBlocks.addr ( 16 * iblk );
    row   = lowerBlocks.addr ( lowerOffsets[iblk] );
    col   = upperBlocks.addr ( upperOffsets[jblk] );

    if ( jblk0 < iblk0 )
    {
      n    =  4 * (iblk  - iblk0);
      row += 16 * (iblk0 - jblk0);
    }
    else
    {
      n    =  4 * (iblk  - jblk0);
      col += 16 * (jblk0 - iblk0);
    }

    a00 = a01 = a02 = a03 = 0.0;
    a10 = a11 = a12 = a13 = 0.0;
    a20 = a21 = a22 = a23 = 0.0;
    a30 = a31 = a32 = a33 = 0.0;

    for ( idx_t i = 0; i < n; i++ )
    {
      a00 += row[0] * col[0];
      a01 += row[0] * col[1];
      a02 += row[0] * col[2];
      a03 += row[0] * col[3];

      a10 += row[1] * col[0];
      a11 += row[1] * col[1];
      a12 += row[1] * col[2];
      a13 += row[1] * col[3];

      a20 += row[2] * col[0];
      a21 += row[2] * col[1];
      a22 += row[2] * col[2];
      a23 += row[2] * col[3];

      a30 += row[3] * col[0];
      a31 += row[3] * col[1];
      a32 += row[3] * col[2];
      a33 += row[3] * col[3];

      row += 4;
      col += 4;
    }

    col[0]  -= a00;
    col[1]  -= a01;
    col[2]  -= a02;
    col[3]  -= a03;

    a10     += piv[4]  * col[0];
    a11     += piv[4]  * col[1];
    a12     += piv[4]  * col[2];
    a13     += piv[4]  * col[3];

    a20     += piv[8]  * col[0];
    a21     += piv[8]  * col[1];
    a22     += piv[8]  * col[2];
    a23     += piv[8]  * col[3];

    a30     += piv[12] * col[0];
    a31     += piv[12] * col[1];
    a32     += piv[12] * col[2];
    a33     += piv[12] * col[3];

    col[4]  -= a10;
    col[5]  -= a11;
    col[6]  -= a12;
    col[7]  -= a13;

    a20     += piv[9]  * col[4];
    a21     += piv[9]  * col[5];
    a22     += piv[9]  * col[6];
    a23     += piv[9]  * col[7];

    a30     += piv[13] * col[4];
    a31     += piv[13] * col[5];
    a32     += piv[13] * col[6];
    a33     += piv[13] * col[7];

    col[8]  -= a20;
    col[9]  -= a21;
    col[10] -= a22;
    col[11] -= a23;

    a30     += piv[14] * col[8];
    a31     += piv[14] * col[9];
    a32     += piv[14] * col[10];
    a33     += piv[14] * col[11];

    col[12] -= a30;
    col[13] -= a31;
    col[14] -= a32;
    col[15] -= a33;

    flops   += (24.0 + 16.0 * (double) n);
  }

  return flops;
}


//=======================================================================
//   class Worker_
//=======================================================================


class SkylineSolver::Worker_ : public Thread
{
 public:

  enum                      Status
  {
                              DEAD,
                              IDLE,
                              WAITING,
                              WORKING
  };

  enum                      Command
  {
                              NO_COMMAND,
                              ABORT_JOB,
                              START_FACTOR,
                              FACTOR_4ROWS
  };


  explicit                  Worker_

    ( const Ref<Data_>&       data );

  virtual void              run           () override;
  void                      abortJob      ();
  inline void               finishJob     ();
  inline void               startFactor   ();
  inline void               factor4Rows   ();


 private:

  void                      factor_       ();

  inline void               setStatus_

    ( Status                  stat );

  inline int                getCommand_   ();

  inline void               setCommand_

    ( Command                 cmd );


 private:

  Ref<Data_>                data_;
  Monitor                   monitor_;

  Status                    status_;
  Command                   command_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


SkylineSolver::Worker_::Worker_ ( const Ref<Data_>& data ) :

  data_ ( data )

{
  status_  = DEAD;
  command_ = NO_COMMAND;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


void SkylineSolver::Worker_::run ()
{
  allowCancel ( true );

  while ( true )
  {
    setStatus_ ( IDLE );

    if ( getCommand_() == START_FACTOR )
    {
      factor_ ();
    }
  }
}


//-----------------------------------------------------------------------
//   abortJob
//-----------------------------------------------------------------------


void SkylineSolver::Worker_::abortJob ()
{
  Lock<Monitor>  lock ( monitor_ );

  command_ = ABORT_JOB;

  monitor_.notify ();

  while ( status_ == WORKING )
  {
    monitor_.waitNoCancel ();
  }
}


//-----------------------------------------------------------------------
//   finishJob
//-----------------------------------------------------------------------


inline void SkylineSolver::Worker_::finishJob ()
{
  Lock<Monitor>  lock ( monitor_ );

  while ( status_ == WORKING || command_ != NO_COMMAND )
  {
    monitor_.wait ();
  }
}


//-----------------------------------------------------------------------
//   startFactor
//-----------------------------------------------------------------------


inline void SkylineSolver::Worker_::startFactor ()
{
  setCommand_ ( START_FACTOR );
}


//-----------------------------------------------------------------------
//   factor4Rows
//-----------------------------------------------------------------------


inline void SkylineSolver::Worker_::factor4Rows ()
{
  setCommand_ ( FACTOR_4ROWS );
}


//-----------------------------------------------------------------------
//   factor_
//-----------------------------------------------------------------------


void SkylineSolver::Worker_::factor_ ()
{
  const idx_t  bcount = data_->blockCount ();

  for ( idx_t iblk = 0; iblk < bcount; iblk++ )
  {
    int  cmd = getCommand_ ();

    if ( cmd == ABORT_JOB )
    {
      return;
    }

    JEM_ASSERT ( cmd == FACTOR_4ROWS );

    data_->factor4Rows ( iblk );
  }
}


//-----------------------------------------------------------------------
//   setStatus_
//-----------------------------------------------------------------------


inline void SkylineSolver::Worker_::setStatus_ ( Status stat )
{
  Lock<Monitor>  lock ( monitor_ );

  status_ = stat;

  monitor_.notify ();
}


//-----------------------------------------------------------------------
//   getCommand_
//-----------------------------------------------------------------------


inline int SkylineSolver::Worker_::getCommand_ ()
{
  Lock<Monitor>  lock ( monitor_ );

  Command        cmd;

  while ( command_ == NO_COMMAND )
  {
    status_ = WAITING;

    monitor_.notify ();
    monitor_.wait   ();
  }

  cmd      = command_;
  command_ = NO_COMMAND;
  status_  = WORKING;

  monitor_.notify ();

  return cmd;
}


//-----------------------------------------------------------------------
//   setCommand_
//-----------------------------------------------------------------------


inline void SkylineSolver::Worker_::setCommand_ ( Command cmd )
{
  Lock<Monitor>  lock ( monitor_ );

  while ( command_ != NO_COMMAND )
  {
    monitor_.wait ();
  }

  command_ = cmd;

  monitor_.notify ();
}


//=======================================================================
//   class SkylineSolver
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SkylineSolver::TYPE_NAME    = "Skyline";

const idx_t  SkylineSolver::MAX_ITER     = 10;

const int    SkylineSolver::NEW_VALUES_  = 1 << 0;
const int    SkylineSolver::NEW_STRUCT_  = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SkylineSolver::SkylineSolver

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super ( name )

{
  using jem::util::connect;
  using jive::util::joinNames;

  JEM_PRECHECK ( matrix );

  const String  conName = joinNames ( myName_, "constrainer" );


  if ( ! matrix->hasExtension<SparseMatrixExt>() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      matrix->getContext () +
      " does not implement the sparse matrix extension"
    );
  }

  if ( matrix->isDistributed() )
  {
    throw IllegalInputException (
      getContext (),
      getContext () + " does not support distributed matrices"
    );
  }

  if ( ! cons )
  {
    conman_ =

      newInstance<DummyConstrainer> ( conName, matrix );
  }
  else
  {
    conman_ =

      newInstance<StdConstrainer>   ( conName, cons, matrix );
  }

  matrix_    = conman_->getOutputMatrix ();
  mode_      = 0;
  small_     = ZERO_THRESHOLD;
  precision_ = PRECISION;
  options_   = REORDER;
  maxZeroes_ = 0;
  iiter_     = 0;
  error_     = 0.0;
  events_    = ~0x0;
  started_   = 0;

  connect ( matrix_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( matrix_->newStructEvent, this, & Self::structChanged_ );
}


SkylineSolver::~SkylineSolver ()
{
  if ( worker_ )
  {
    worker_->cancel ();
    worker_->join   ();
  }
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SkylineSolver::start ()
{
  conman_->update ();

  if ( ! started_ )
  {
    matrix_->resetEvents ();
  }

  if ( events_ )
  {
    update_ ();
  }

  started_++;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void SkylineSolver::finish ()
{
  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SkylineSolver::clear ()
{
  JEM_PRECHECK ( ! started_ );

  if ( worker_ )
  {
    worker_->cancel ();
    worker_->join   ();
  }

  data_   = nullptr;
  events_ = ~0x0;
}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void SkylineSolver::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  using jem::dot;
  using jem::max;

  JEM_PRECHECK ( lhs.size() == rhs.size() );

  SolverScope  scope    ( *this );

  const idx_t  dofCount = data_->matrixSize ();

  Vector       du;
  Vector       r;
  Vector       u;
  Vector       f;

  double       rscale;


  if ( lhs.size() != dofCount )
  {
    util::sizeError ( getContext(),
                      "lhs vector", lhs.size(), dofCount );
  }

  if ( mode_ & PRECON_MODE )
  {
    Matrix  vbuf ( dofCount, 2 );

    r .ref ( vbuf[0] );
    du.ref ( vbuf[1] );
    u .ref ( lhs );
    f .ref ( rhs );
  }
  else
  {
    Matrix  vbuf ( dofCount, 4 );

    r .ref ( vbuf[0] );
    du.ref ( vbuf[1] );
    u .ref ( vbuf[2] );
    f .ref ( vbuf[3] );

    conman_->getRhs  ( f, rhs );
    conman_->initLhs ( u, lhs );
  }

  iiter_ = 0;
  error_ = 0.0;
  rscale = std::sqrt ( dot( f, f ) );

  if ( Float::isNaN( rscale ) )
  {
    throw ArithmeticException (
      getContext (),
      "invalid norm of right-hand vector: NaN"
    );
  }

  if ( Float::isTiny( rscale ) )
  {
    u = 0.0;

    if ( ! (mode_ & PRECON_MODE) )
    {
      conman_->getLhs ( lhs, u );
    }

    return;
  }

  rscale = 1.0 / rscale;

  matrix_->matmul ( r, u );

  r = f - r;

  while ( iiter_ < MAX_ITER )
  {
    fbsub_ ( du, r );

    u  += du;

    matrix_->matmul ( r, u );

    r = f - r;

    iiter_++;
    error_ = rscale * std::sqrt ( dot( r, r ) );

    solveEvent.emit ( error_, *this );

    if ( Float::isNaN( error_ ) )
    {
      throw ArithmeticException (
        getContext (),
        "invalid norm of residual vector: NaN"
      );
    }

    if ( error_ <= precision_ || error_ > 1.0e5 )
    {
      break;
    }
  }

  if ( (error_ > max( 1.0, precision_ )) ||
       (error_ > precision_ && ! (mode_ & LENIENT_MODE)) )
  {
    throw SolverException (
      getContext     (),
      String::format ( "residual norm too large: %e", error_ )
    );
  }

  if ( ! (mode_ & PRECON_MODE) )
  {
    conman_->getLhs ( lhs, u );
  }
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void SkylineSolver::getInfo ( const Properties& info ) const
{
  double  memUsage = 0.0;
  idx_t   dofCount = matrix_->size (0);

  if ( data_ )
  {
    memUsage = data_->getMemUsage ();
  }

  Super::getInfo ( info );

  info.set ( SolverInfo::TYPE_NAME,  TYPE_NAME );
  info.set ( SolverInfo::MEM_USAGE,  memUsage  );
  info.set ( SolverInfo::RESIDUAL,   error_    );
  info.set ( SolverInfo::ITER_COUNT, iiter_    );
  info.set ( SolverInfo::DOF_COUNT,  dofCount  );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SkylineSolver::configure ( const Properties& props )
{
  using jem::util::findBool;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    Options  options = options_;

    findBool ( options, REORDER,
               myProps, PropNames::REORDER );
    findBool ( options, USE_THREADS,
               myProps, PropNames::USE_THREADS );
    findBool ( options, PRINT_PIVOTS,
               myProps, PropNames::PRINT_PIVOTS );

    if ( options != options_ )
    {
      setOptions ( options );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SkylineSolver::getConfig ( const Properties& conf ) const
{
  using jem::util::setBool;

  Properties  myConf = conf.makeProps ( myName_ );

  Super::getConfig ( conf );

  setBool ( myConf,   PropNames::REORDER,
            options_, REORDER );
  setBool ( myConf,   PropNames::USE_THREADS,
            options_, USE_THREADS );
  setBool ( myConf,   PropNames::PRINT_PIVOTS,
            options_, PRINT_PIVOTS );
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void SkylineSolver::setMode ( int mode )
{
  mode_ = mode;
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int SkylineSolver::getMode () const
{
  return mode_;
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void SkylineSolver::setPrecision ( double eps )
{
  JEM_PRECHECK ( eps >= 0.0 );

  precision_ = eps;
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double SkylineSolver::getPrecision () const
{
  return precision_;
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* SkylineSolver::getMatrix () const
{
  return conman_->getInputMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* SkylineSolver::getConstraints () const
{
  return conman_->getConstraints ();
}


//-----------------------------------------------------------------------
//   getNullSpace
//-----------------------------------------------------------------------


void SkylineSolver::getNullSpace ( Matrix& nspace )
{
  SolverScope  scope     ( *this );

  Data_&       d         = *data_;

  const idx_t  dofCount  = d.matrixSize      ();
  const idx_t  zeroCount = d.zeroPivots.size ();


  nspace.resize ( dofCount, zeroCount );

  if ( zeroCount > 0 )
  {
    Vector  v ( dofCount );

    v = 0.0;

    for ( idx_t j = 0; j < zeroCount; j++ )
    {
      idx_t  irow = d.zeroPivots[j];

      v[irow] = 1.0;

      fbsub_ ( nspace[j], v );

      v[irow] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   setZeroThreshold
//-----------------------------------------------------------------------


void SkylineSolver::setZeroThreshold ( double eps )
{
  small_ = eps;

  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   getZeroThreshold
//-----------------------------------------------------------------------


double SkylineSolver::getZeroThreshold () const
{
  return small_;
}


//-----------------------------------------------------------------------
//   setMaxZeroPivots
//-----------------------------------------------------------------------


void SkylineSolver::setMaxZeroPivots ( idx_t n )
{
  maxZeroes_ = n;

  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   getMaxZeroPivots
//-----------------------------------------------------------------------


idx_t SkylineSolver::getMaxZeroPivots () const
{
  return maxZeroes_;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void SkylineSolver::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void SkylineSolver::setOptions ( Options options )
{
  options_ = options;

  setEvents_ ( NEW_STRUCT_ );

  // Kill the worker thread if required.

  if ( ! (options & USE_THREADS) && worker_ )
  {
    worker_->cancel ();
    worker_->join   ();

    worker_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> SkylineSolver::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<AbstractMatrix>  mat;
  Ref<Constraints>     cons;

  params.find ( mat,  SolverParams::MATRIX      );
  params.find ( cons, SolverParams::CONSTRAINTS );

  if ( mat && mat->hasExtension<SparseMatrixExt>() )
  {
    return newInstance<Self> ( name, mat, cons );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SkylineSolver::declare ()
{
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   fbsub_
//-----------------------------------------------------------------------


void SkylineSolver::fbsub_

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  Data_&         d = *data_;

  const idx_t*   JEM_RESTRICT  iperm = d.iperm   .addr ();
  const double*  JEM_RESTRICT  scale = d.rowScale.addr ();

  const double*  JEM_RESTRICT  piv;
  const double*  JEM_RESTRICT  row;
  const double*  JEM_RESTRICT  col;
  double*        JEM_RESTRICT  tmp;

  const idx_t    msize  = d.matrixSize ();
  const idx_t    bcount = d.blockCount ();

  double         a0, a1, a2, a3;

  idx_t          iblk0;
  idx_t          jblk0;
  idx_t          n;


  // Scale and permute the right-hand side vector.

  tmp = d.tempVector.addr ();

JEM_IVDEP

  for ( idx_t i = 0; i < msize; i++ )
  {
    tmp[iperm[i]] = rhs[i] * scale[i];
  }

  // Forward substitution with the lower factor.

  for ( idx_t iblk = 0; iblk < bcount; iblk++ )
  {
    jblk0 = d.startCols[iblk];
    n     = 4_idx * (iblk - jblk0);

    tmp   = d.tempVector .addr (  4 * jblk0 );
    piv   = d.pivotBlocks.addr ( 16 * iblk );
    row   = d.lowerBlocks.addr ( d.lowerOffsets[iblk] );

    a0 = a1 = a2 = a3 = 0.0;

    for ( idx_t j = 0; j < n; j++ )
    {
      a0  += row[0] * tmp[j];
      a1  += row[1] * tmp[j];
      a2  += row[2] * tmp[j];
      a3  += row[3] * tmp[j];

      row += 4;
    }

    tmp    += n;

    tmp[0] -= a0;
    a1     += tmp[0] * piv[4];
    a2     += tmp[0] * piv[8];
    a3     += tmp[0] * piv[12];
    tmp[1] -= a1;
    a2     += tmp[1] * piv[9];
    a3     += tmp[1] * piv[13];
    tmp[2] -= a2;
    a3     += tmp[2] * piv[14];
    tmp[3] -= a3;
  }

  if ( Thread::cancelled() )
  {
    throw CancelledException (
      getContext (),
      "forward/backward substitution cancelled"
    );
  }

  // Backward substitution with the upper factor.

  for ( idx_t jblk = bcount - 1; jblk >= 0; jblk-- )
  {
    iblk0 = d.startRows[jblk];
    n     = 4_idx * (jblk - iblk0);

    tmp   = d.tempVector .addr (  4 * jblk );
    piv   = d.pivotBlocks.addr ( 16 * jblk );
    col   = d.upperBlocks.addr ( d.upperOffsets[jblk] );

    tmp[3] = a3 = piv[15] *  tmp[3];
    tmp[2] = a2 = piv[10] * (tmp[2] - a3 * piv[11]);
    tmp[1] = a1 = piv[5]  * (tmp[1] - a3 * piv[7] - a2 * piv[6]);
    tmp[0] = a0 = piv[0]  * (tmp[0] - a3 * piv[3] - a2 * piv[2] -
                                      a1 * piv[1]);

    tmp   -= n;

JEM_IVDEP

    for ( idx_t i = 0; i < n; i++ )
    {
      tmp[i] -= (a0 * col[0] + a1 * col[1] +
                 a2 * col[2] + a3 * col[3]);

      col    += 4;
    }
  }

  // Store the permuted solution in the left-hand vector.

  tmp = d.tempVector.addr ();

JEM_IVDEP

  for ( idx_t i = 0; i < msize; i++ )
  {
    lhs[i] = tmp[iperm[i]];
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SkylineSolver::update_ ()
{
  AbstractMatrix::Shape  sh = matrix_->shape ();

  bool  result = false;

  if ( sh[0] != sh[1] )
  {
    util::nonSquareMatrixError ( getContext(), sh );
  }

  if ( ! data_ )
  {
    data_ = newInstance<Data_> ();
  }

  if ( (options_ & USE_THREADS) && (worker_ == nullptr) )
  {
    print ( System::debug( myName_ ), myName_,
            " : spawning worker thread ...\n" );

    worker_ = newInstance<Worker_> ( data_ );

    worker_->start ();
  }

  // Make sure the worker thread is not touching the solver data.

  if ( worker_ )
  {
    worker_->abortJob ();
  }

  if ( sh[0] != data_->matrixSize() )
  {
    events_ |= NEW_STRUCT_;
  }

  try
  {
    if ( events_ & NEW_STRUCT_ )
    {
      events_ |= NEW_VALUES_;

      initSkyline_ ();
    }

    loadSkyline_ ();

    result = factorSkyline_ ();
  }
  catch ( ... )
  {
    if ( worker_ )
    {
      worker_->abortJob ();
    }

    throw;
  }

  if ( ! result )
  {
    String  msg;

    if ( maxZeroes_ == 0 )
    {
      msg =

        "maybe singular matrix (zero pivot); "
        "try the `SparseLU\' solver to be sure";
    }
    else
    {
      msg = String::format (
        "too many (%d) zero pivots encountered",
        data_->zeroPivots.size ()
      );
    }

    throw SolverException ( getContext(), msg );
  }

  matrix_->resetEvents ();

  events_ = 0;
}


//-----------------------------------------------------------------------
//   initSkyline_
//-----------------------------------------------------------------------


void SkylineSolver::initSkyline_ ()
{
  using jem::min;
  using jem::max;
  using jem::iarray;

  Data_&        d         = *data_;

  SparseStruct  matStruct = matrix_ ->

    getExtension<SparseMatrixExt>()  -> getStructure   ();

  IdxVector    rowOffsets = matStruct.getRowOffsets    ();
  IdxVector    colIndices = matStruct.getColumnIndices ();

  const idx_t  msize      = matStruct.size (0);
  const idx_t  bcount     = (msize + 3) / 4;

  idx_t        lowerSize  = 0;
  idx_t        upperSize  = 0;
  idx_t        bandwidth  = 0;


  d.resize ( msize );

  if ( options_ & REORDER )
  {
    print ( System::debug( myName_ ), myName_,
            " : re-ordering matrix ...\n" );

    d.reorder ( matStruct );
  }
  else
  {
    d.iperm = iarray ( msize );
  }

  // Determine the starting row/column indices.

  d.startRows = iarray ( bcount );
  d.startCols = iarray ( bcount );

  for ( idx_t irow = 0; irow < msize; irow++ )
  {
    idx_t  iblk = d.iperm[irow] / 4_idx;
    idx_t  n    = rowOffsets[irow + 1];

    for ( idx_t j = rowOffsets[irow]; j < n; j++ )
    {
      idx_t  jblk = d.iperm[colIndices[j]] / 4_idx;

      d.startCols[iblk] = min ( d.startCols[iblk], jblk );
      d.startRows[jblk] = min ( d.startRows[jblk], iblk );
    }
  }

  // Initialize the lower and upper offset arrays.

  for ( idx_t iblk = 0; iblk < bcount; iblk++ )
  {
    idx_t  lowerWidth = iblk - d.startCols[iblk];
    idx_t  upperWidth = iblk - d.startRows[iblk];

    d.lowerOffsets[iblk] = lowerSize;
    d.upperOffsets[iblk] = upperSize;

    lowerSize += 16_idx * lowerWidth;
    upperSize += 16_idx * upperWidth;

    bandwidth  = max ( bandwidth, lowerWidth, upperWidth );
  }

  d.lowerOffsets[bcount] = lowerSize;
  d.upperOffsets[bcount] = upperSize;

  d.lowerBlocks.resize ( lowerSize );
  d.upperBlocks.resize ( upperSize );

  print ( System::debug( myName_ ), myName_,
          " : half-bandwidth = ", 4 * bandwidth, endl );
}


//-----------------------------------------------------------------------
//   loadSkyline_
//-----------------------------------------------------------------------


void SkylineSolver::loadSkyline_ ()
{
  using jem::max;
  using jem::sum;
  using jem::isTiny;

  Data_&        d       = *data_;

  SparseMatrix  matrix  = matrix_ ->

    getExtension<SparseMatrixExt>() -> toSparseMatrix ();

  IdxVector     rowOffsets = matrix.getRowOffsets    ();
  IdxVector     colIndices = matrix.getColumnIndices ();
  Vector        matValues  = matrix.getValues        ();

  const idx_t   msize      = d.matrixSize ();
  const idx_t   bcount     = d.blockCount ();


  if ( Float::isNaN( sum( matValues ) ) )
  {
    throw ArithmeticException (
      getContext (),
      matrix_->getContext() +
      " contains invalid data: NaN"
    );
  }

  d.zeroPivots.clear ();

  d.rowScale    = 0.0;
  d.tempVector  = 0.0;
  d.pivotBlocks = 0.0;
  d.lowerBlocks = 0.0;
  d.upperBlocks = 0.0;

  // Determine the row scale values.

  for ( idx_t irow = 0; irow < msize; irow++ )
  {
    double  t = -1.0;
    double  s =  d.rowScale[irow];
    idx_t   n =  rowOffsets[irow + 1];

    for ( idx_t j = rowOffsets[irow]; j < n; j++ )
    {
      s = max ( s, std::fabs( matValues[j] ) );

      if ( irow == colIndices[j] )
      {
        t = std::fabs ( matValues[j] );
      }
    }

    // Use the diagonal entry if it is non-zero, and the maximum row
    // value otherwise.

    if ( t > 0.0 )
    {
      d.rowScale[irow] = t;
    }
    else
    {
      d.rowScale[irow] = s;
    }
  }

  // Invert the scale values.

  for ( idx_t irow = 0; irow < msize; irow++ )
  {
    double  s = d.rowScale[irow];

    if ( isTiny( s ) )
    {
      s = 1.0;
    }
    else
    {
      s = 1.0 / s;
    }

    d.rowScale[irow] = s;
  }

  // Store the scaled matrix values in the skyline structure.

  for ( idx_t irow = 0; irow < msize; irow++ )
  {
    idx_t   iblk = d.iperm[irow] / 4;
    idx_t   ipos = d.iperm[irow] % 4;

    double  s    = d.rowScale[irow];
    idx_t   n    = rowOffsets[irow + 1];

    for ( idx_t j = rowOffsets[irow]; j < n; j++ )
    {
      double  mval = s * matValues[j];

      idx_t   jcol = colIndices[j];
      idx_t   jblk = d.iperm[jcol] / 4_idx;
      idx_t   jpos = d.iperm[jcol] % 4_idx;

      if ( jblk < iblk )
      {
        idx_t  k = 16_idx * (jblk - d.startCols[iblk]);

        if ( k < 0 )
        {
          matStructError_ ( irow, jcol );
        }

        k += d.lowerOffsets[iblk] + ipos + 4_idx * jpos;

        d.lowerBlocks[k] = mval;
      }
      else if ( jblk > iblk )
      {
        idx_t  k = 16_idx * (iblk - d.startRows[jblk]);

        if ( k < 0 )
        {
          matStructError_ ( irow, jcol );
        }

        k += d.upperOffsets[jblk] + 4_idx * ipos + jpos;

        d.upperBlocks[k] = mval;
      }
      else
      {
        idx_t  k = 16_idx * iblk + 4_idx * ipos + jpos;

        d.pivotBlocks[k] = mval;
      }
    }
  }

  // Make sure that the final pivot values outside the matrix
  // structure are non-zero.

  if ( bcount > 0 )
  {
    idx_t  k = 16_idx * (bcount - 1);

    for ( idx_t i = msize % 4_idx; i < 4; i++ )
    {
      d.pivotBlocks[k + 4 * i + i] = 1.0;
    }
  }
}


//-----------------------------------------------------------------------
//   factorSkyline_
//-----------------------------------------------------------------------


bool SkylineSolver::factorSkyline_ ()
{
  using jem::min;

  Data_&       d      = *data_;
  Worker_*     worker = 0;

  double*      JEM_RESTRICT  piv = 0;
  double*      JEM_RESTRICT  row = 0;
  double*      JEM_RESTRICT  col = 0;

  const idx_t  bcount  = d.blockCount ();

  double       flops;

  double       a00, a01, a02, a03;
  double       a10, a11, a12, a13;
  double       a20, a21, a22, a23;
  double       a30, a31, a32, a33;

  idx_t        maxZeroes;
  idx_t        iblk0;
  idx_t        n;


  if ( options_ & USE_THREADS )
  {
    worker = worker_.get ();
  }

  maxZeroes = maxZeroes_;

  if ( (maxZeroes) < 0 || (mode_ & LENIENT_MODE) )
  {
    maxZeroes = d.matrixSize ();
  }

  factorEvent.emit ( 0, *this );

  // Factor the matrix using Crout's algorithm.

  if ( worker )
  {
    worker->startFactor ();
  }

  flops = 0.0;

  for ( idx_t iblk = 0; iblk < bcount; iblk++ )
  {
    idx_t  irow = 4_idx * iblk;

    // Compute the next block of the lower/upper factor.

    if ( worker )
    {
      worker->factor4Rows ();
    }
    else
    {
      flops += d.factor4Rows  ( iblk );
    }

    flops += d.factor4Columns ( iblk );

    if ( worker )
    {
      worker_->finishJob ();
    }

    iblk0  = min ( d.startRows[iblk], d.startCols[iblk] );
    n      = 4_idx * (iblk - iblk0);

    piv = d.pivotBlocks.addr ( 16 * iblk );
    row = d.lowerBlocks.addr ( d.lowerOffsets[iblk] );
    col = d.upperBlocks.addr ( d.upperOffsets[iblk] );

    a00 = a01 = a02 = a03 = 0.0;
    a10 = a11 = a12 = a13 = 0.0;
    a20 = a21 = a22 = a23 = 0.0;
    a30 = a31 = a32 = a33 = 0.0;

    // Compute the new pivots.

    for ( idx_t i = 0; i < n; i++ )
    {
      a00 += row[0] * col[0];
      a01 += row[0] * col[1];
      a02 += row[0] * col[2];
      a03 += row[0] * col[3];

      a10 += row[1] * col[0];
      a11 += row[1] * col[1];
      a12 += row[1] * col[2];
      a13 += row[1] * col[3];

      a20 += row[2] * col[0];
      a21 += row[2] * col[1];
      a22 += row[2] * col[2];
      a23 += row[2] * col[3];

      a30 += row[3] * col[0];
      a31 += row[3] * col[1];
      a32 += row[3] * col[2];
      a33 += row[3] * col[3];

      row += 4;
      col += 4;
    }

    piv[0]  = calcPivot_ ( irow, piv[0] - a00 );

    piv[1]  =  piv[1]  - a01;
    piv[2]  =  piv[2]  - a02;
    piv[3]  =  piv[3]  - a03;
    piv[4]  = (piv[4]  - a10) * piv[0];
    piv[8]  = (piv[8]  - a20) * piv[0];
    piv[12] = (piv[12] - a30) * piv[0];

    a11    += piv[4]  * piv[1];
    a12    += piv[4]  * piv[2];
    a13    += piv[4]  * piv[3];
    a21    += piv[8]  * piv[1];
    a22    += piv[8]  * piv[2];
    a23    += piv[8]  * piv[3];
    a31    += piv[12] * piv[1];
    a32    += piv[12] * piv[2];
    a33    += piv[12] * piv[3];

    piv[5]  = calcPivot_ ( irow + 1, piv[5] - a11 );

    piv[6]  =  piv[6]  - a12;
    piv[7]  =  piv[7]  - a13;
    piv[9]  = (piv[9]  - a21) * piv[5];
    piv[13] = (piv[13] - a31) * piv[5];

    a22    += piv[9]  * piv[6];
    a23    += piv[9]  * piv[7];
    a32    += piv[13] * piv[6];
    a33    += piv[13] * piv[7];

    piv[10] = calcPivot_ ( irow + 2, piv[10] - a22 );

    piv[11] =  piv[11] - a23;
    piv[14] = (piv[14] - a32) * piv[10];

    a33    += piv[14] * piv[11];

    piv[15] = calcPivot_ ( irow + 3, piv[15] - a33 );

    flops  += (20.0 + 16.0 * (double) n);

    if ( flops > 1.0e7 )
    {
      flops = 0.0;

      if ( Thread::cancelled() )
      {
        throw CancelledException (
          getContext (),
          "LU factorization cancelled"
        );
      }

      if ( factorEvent.isConnected() )
      {
        double  progress = 100.0 * (double) (iblk + 1) / (double) bcount;

        factorEvent.emit ( (idx_t) progress, *this );
      }
    }

    if ( d.zeroPivots.size() > maxZeroes )
    {
      return false;
    }
  }

  factorEvent.emit ( 100, *this );

  return true;
}


//-----------------------------------------------------------------------
//   calcPivot_
//-----------------------------------------------------------------------


double SkylineSolver::calcPivot_

  ( idx_t   irow,
    double  piv )

{
  using jem::isTiny;

  if ( std::fabs( piv ) < small_ )
  {
    idx_t  jrow = data_->invPerm[irow];

    if ( zeroPivotEvent.isConnected() )
    {
      zeroPivotEvent.emit ( piv, *this );
    }

    data_->zeroPivots.pushBack ( jrow );

    if ( isTiny( piv ) )
    {
      piv  = std::sqrt ( small_ );
    }
    else
    {
      piv *= std::sqrt ( small_ ) / std::fabs ( piv );
    }
  }
  else
  {
    if ( options_ & PRINT_PIVOTS )
    {
      print ( System::debug( myName_ ),
              "  pivot (row,value): ",
              data_->invPerm[irow], '\t', piv, endl );
    }

    piv = 1.0 / piv;
  }

  return piv;
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SkylineSolver::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SkylineSolver::structChanged_ ()
{
  setEvents_ ( NEW_STRUCT_ );
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void SkylineSolver::setEvents_ ( int events )
{
  if ( started_ )
  {
    throw jem::IllegalOperationException (
      getContext (),
      "matrix changed while solving a linear system of equations"
    );
  }

  events_ |= events;
}


//-----------------------------------------------------------------------
//   matStructError_
//-----------------------------------------------------------------------


void SkylineSolver::matStructError_

  ( idx_t  irow,
    idx_t  jcol ) const

{
  String  matName = matrix_->getName ();

  throw IllegalInputException (
    getContext     (),
    String::format (
      "matrix `%s\' has an invalid structure; "
      "matrix element (%d,%d) is located outside the "
      "solver skyline",
      matName,
      irow, jcol
    )
  );
}


JIVE_END_PACKAGE( solver )
