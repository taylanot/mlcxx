
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Flex.h>
#include <jem/numeric/sparse/Reorder.h>
#include <jem/numeric/sparse/SparseLU.h>
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
#include <jive/solver/SparseLU.h>


JEM_DEFINE_CLASS( jive::solver::SparseLU );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::util::Flex;
using jive::algebra::SparseMatrixExt;


//=======================================================================
//   class SparseLU::Data_
//=======================================================================


class SparseLU::Data_ : public jem::Collectable
{
 public:

  inline idx_t              matrixSize    () const;
  double                    getMemUsage   () const;


 public:

  jem::numeric::SparseLU    solver;

  IdxVector                 rowPerm;
  IdxVector                 colPerm;
  Flex<idx_t>               zeroPivots;

};


//-----------------------------------------------------------------------
//   matrixSize
//-----------------------------------------------------------------------


inline idx_t SparseLU::Data_::matrixSize () const
{
  return rowPerm.size ();
}


//-----------------------------------------------------------------------
//   getMemUsage
//-----------------------------------------------------------------------


double SparseLU::Data_::getMemUsage () const
{
  double  isize  = (double) sizeof(idx_t);
  double  musage = 0.0;

  musage += solver.getMemUsage ();
  musage += isize * (double) rowPerm.size ();
  musage += isize * (double) colPerm.size ();

  return musage;
}


//=======================================================================
//   class SparseLU
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*   SparseLU::TYPE_NAME          = "SparseLU";
const double  SparseLU::PIVOT_THRESHOLD    = 0.1;
const int     SparseLU::MAX_ITER           = 10;

const char*   SparseLU::REORDER_METHODS[3] =
{
  "None",
  "Matrix",
  "Columns"
};

const int     SparseLU::NEW_VALUES_        = 1 << 0;
const int     SparseLU::NEW_STRUCT_        = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SparseLU::SparseLU

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<Constraints>     cons ) :

    Super ( name )

{
  using jem::util::connect;
  using jive::util::joinNames;

  JEM_PRECHECK ( matrix );

  String  conName = joinNames ( myName_, "constrainer" );


  if ( ! matrix->hasExtension<SparseMatrixExt>() )
  {
    throw jem::IllegalArgumentException (
      JEM_FUNC,
      matrix->getContext() +
      " does not implement the sparse matrix extension"
    );
  }

  if ( matrix->isDistributed() )
  {
    throw jem::IllegalInputException (
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
  smallPiv_  = PIVOT_THRESHOLD;
  precision_ = PRECISION;
  reorder_   = REORDER_MATRIX;
  options_   = 0;
  maxZeroes_ = 0;
  iiter_     = 0;
  error_     = 0.0;
  events_    = ~0x0;
  started_   = 0;

  connect ( matrix_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( matrix_->newStructEvent, this, & Self::structChanged_ );
}


SparseLU::~SparseLU ()
{}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void SparseLU::start ()
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


void SparseLU::finish ()
{
  if ( started_ )
  {
    started_--;
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SparseLU::clear ()
{
  JEM_PRECHECK ( ! started_ );

  data_   = nullptr;
  events_ = ~0x0;
}


//-----------------------------------------------------------------------
//   improve
//-----------------------------------------------------------------------


void SparseLU::improve

  ( const Vector&  lhs,
    const Vector&  rhs )

{
  using jem::dot;
  using jem::max;
  using jem::Float;
  using jem::ArithmeticException;

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
    data_->solver.solve ( du, r );

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


void SparseLU::getInfo ( const Properties& info ) const
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


void SparseLU::configure ( const Properties& props )
{
  using jem::util::findBool;

  Super::configure ( props );

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      reord;

    myProps.find ( smallPiv_, PropNames::PIVOT_THRESHOLD,
                   0.0, 1.0 );

    if ( myProps.find( reord, PropNames::REORDER ) )
    {
      int  i;

      for ( i = 0; i < 3; i++ )
      {
        if ( reord.equalsIgnoreCase( REORDER_METHODS[i] ) )
        {
          break;
        }
      }

      if ( i >= 3 )
      {
        myProps.propertyError (
          PropNames::REORDER,
          String::format (
            "invalid re-ordering method; "
            "should be `%s\', `%s\' or `%s\'",
            REORDER_METHODS[0],
            REORDER_METHODS[1],
            REORDER_METHODS[2]
          )
        );
      }

      reorder_ = (ReorderMethod) i;
    }

    findBool ( options_, PRINT_PIVOTS,
               myProps,  PropNames::PRINT_PIVOTS );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SparseLU::getConfig ( const Properties& conf ) const
{
  using jem::util::setBool;

  Properties  myConf = conf.makeProps ( myName_ );

  Super::getConfig ( conf );

  myConf.set ( PropNames::PIVOT_THRESHOLD, smallPiv_ );

  myConf.set ( PropNames::REORDER,
               REORDER_METHODS[reorder_] );

  setBool    ( myConf,   PropNames::PRINT_PIVOTS,
               options_, PRINT_PIVOTS );
}


//-----------------------------------------------------------------------
//   setMode
//-----------------------------------------------------------------------


void SparseLU::setMode ( int mode )
{
  mode_ = mode;
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int SparseLU::getMode () const
{
  return mode_;
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


void SparseLU::setPrecision ( double eps )
{
  JEM_PRECHECK ( eps >= 0.0 );

  precision_ = eps;
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


double SparseLU::getPrecision () const
{
  return precision_;
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* SparseLU::getMatrix () const
{
  return conman_->getInputMatrix ();
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* SparseLU::getConstraints () const
{
  return conman_->getConstraints ();
}


//-----------------------------------------------------------------------
//   getNullSpace
//-----------------------------------------------------------------------


void SparseLU::getNullSpace ( Matrix& nspace )
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

      d.solver.solve ( nspace[j], v );

      v[irow] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   setZeroThreshold
//-----------------------------------------------------------------------


void SparseLU::setZeroThreshold ( double eps )
{
  small_ = eps;

  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   getZeroThreshold
//-----------------------------------------------------------------------


double SparseLU::getZeroThreshold () const
{
  return small_;
}


//-----------------------------------------------------------------------
//   setMaxZeroPivots
//-----------------------------------------------------------------------


void SparseLU::setMaxZeroPivots ( idx_t n )
{
  maxZeroes_ = n;

  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   getMaxZeroPivots
//-----------------------------------------------------------------------


idx_t SparseLU::getMaxZeroPivots () const
{
  return maxZeroes_;
}


//-----------------------------------------------------------------------
//   setPivotThreshold
//-----------------------------------------------------------------------


void SparseLU::setPivotThreshold ( double alpha )
{
  smallPiv_ = alpha;

  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   getPivotThreshold
//-----------------------------------------------------------------------


double SparseLU::getPivotThreshold () const
{
  return smallPiv_;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


SparseLU::Options SparseLU::getOptions () const
{
  return options_;
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void SparseLU::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void SparseLU::setOptions ( Options options )
{
  options_ = options;
}


//-----------------------------------------------------------------------
//   getReorderMethod
//-----------------------------------------------------------------------


SparseLU::ReorderMethod SparseLU::getReorderMethod () const
{
  return reorder_;
}


//-----------------------------------------------------------------------
//   setReorderMethod
//-----------------------------------------------------------------------


void SparseLU::setReorderMethod ( ReorderMethod method )
{
  reorder_ = method;

  setEvents_ ( NEW_STRUCT_ );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Solver> SparseLU::makeNew

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


void SparseLU::declare ()
{
  SolverFactory::declare ( "LU",       & makeNew );
  SolverFactory::declare ( TYPE_NAME,  & makeNew );
  SolverFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SparseLU::update_ ()
{
  using jem::iarray;
  using jem::System;
  using jem::numeric::Reorder;

  const String  context    = getContext ();

  SparseMatrixExt*  sx     =

    matrix_->getExtension<SparseMatrixExt> ();

  SparseMatrix      sm     = sx->toSparseMatrix ();

  const idx_t       msize  = sm.size (0);

  BoolVector        mask   ( msize );

  idx_t             mzp, nzp;


  debug_ = & System::debug ( myName_ );

  if ( sm.size(0) != sm.size(1) )
  {
    util::nonSquareMatrixError ( getContext(), sm.shape() );
  }

  if ( ! data_ )
  {
    data_ = newInstance<Data_> ();
  }

  Data_&  d = *data_;

  mzp = maxZeroes_;

  if ( (mzp < 0) || (mode_ & LENIENT_MODE) )
  {
    mzp = msize;
  }

  // Update the solver settings.

  d.solver.setZeroThreshold  ( small_ );
  d.solver.setPivotThreshold ( smallPiv_ );
  d.solver.setMaxZeroPivots  ( mzp );

  connectToSolver_ ();

  if ( events_ & NEW_STRUCT_ )
  {
    d.rowPerm.resize ( msize );

    if      ( reorder_ == REORDER_MATRIX )
    {
      print ( *debug_, myName_, " : re-ordering matrix ...\n" );

      Reorder::superReorder ( d.rowPerm,
                              sm.getStructure (),
                              & Reorder::automa );

      d.colPerm.ref ( d.rowPerm );
    }
    else if ( reorder_ == REORDER_COLUMNS )
    {
      print ( *debug_, myName_, " : re-ordering columns ...\n" );

      d.colPerm.resize ( msize );
      Reorder::colamd  ( d.colPerm, sm.getStructure() );

      d.rowPerm = iarray ( msize );
    }
    else
    {
      d.colPerm.ref ( d.rowPerm );

      d.rowPerm = iarray ( msize );
    }
  }

  print ( *debug_, myName_, " : factoring matrix ...\n" );

  factorEvent.emit   ( 0, *this );
  d.zeroPivots.clear ();

  mask = true;

  nzp  = d.solver.factor ( sm, mask,
                           d.rowPerm,
                           d.colPerm );

  if ( nzp > mzp )
  {
    String  msg;

    if ( mzp == 0 )
    {
      msg = "singular matrix";
    }
    else
    {
      msg = String::format (
        "too many (%d) zero pivots encountered", nzp
      );
    }

    throw SolverException ( context, msg );
  }

  factorEvent.emit ( 100, *this );

  matrix_->resetEvents ();

  events_ = 0;
}


//-----------------------------------------------------------------------
//   connectToSolver_
//-----------------------------------------------------------------------


void SparseLU::connectToSolver_ ()
{
  using jem::util::connect;

  Data_&  d = *data_;


  if ( (options_ & PRINT_PIVOTS) )
  {
    if ( ! d.solver.pivotEvent.isConnected() )
    {
      connect ( d.solver.pivotEvent, this, & Self::pivotHandler_ );
    }
  }
  else
  {
    d.solver.pivotEvent.clear ();
  }

  if ( factorEvent.isConnected() && !
       d.solver.progressEvent.isConnected() )
  {
    connect ( d.solver.progressEvent,
              this, &Self::progressHandler_ );
  }

  if ( ! d.solver.zeroPivotEvent.isConnected() )
  {
    connect ( d.solver.zeroPivotEvent,
              this, &Self::zeroPivotHandler_ );
  }
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SparseLU::valuesChanged_ ()
{
  setEvents_ ( NEW_VALUES_ );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SparseLU::structChanged_ ()
{
  setEvents_ ( NEW_STRUCT_ );
}


//-----------------------------------------------------------------------
//   setEvents_
//-----------------------------------------------------------------------


void SparseLU::setEvents_ ( int events )
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
//   progressHandler_
//-----------------------------------------------------------------------


void SparseLU::progressHandler_ ( idx_t jcol )
{
  const idx_t  msize = data_->matrixSize ();

  if ( msize <= 1 )
  {
    factorEvent.emit ( 1, * this );
  }
  else
  {
    idx_t  done = (idx_t) ((100.0 * (double) jcol) /
                                    (double) (msize - 1));

    factorEvent.emit ( done, *this );
  }
}


//-----------------------------------------------------------------------
//   pivotHandler_
//-----------------------------------------------------------------------


void SparseLU::pivotHandler_ ( idx_t irow, double pivot )
{
  print ( *debug_, "  pivot (row,value): ",
          irow, '\t', pivot, '\n' );
}


//-----------------------------------------------------------------------
//   zeroPivotHandler_
//-----------------------------------------------------------------------


void SparseLU::zeroPivotHandler_ ( idx_t irow, double pivot )
{
  data_->zeroPivots.pushBack ( irow );

  zeroPivotEvent.emit ( pivot, *this );
}


JIVE_END_PACKAGE( solver )
