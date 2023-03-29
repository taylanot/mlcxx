
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/None.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/DenseTable.h>
#include <jive/util/SparseTable.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/solver/declare.h>
#include <jive/solver/Constrainer.h>
#include <jive/solver/EigenSolver.h>
#include <jive/solver/EigenSolverParams.h>
#include <jive/solver/EigenSolverFactory.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/utilities.h>
#include <jive/implict/Names.h>
#include <jive/implict/Globdat.h>
#include <jive/implict/EigensolveModule.h>


JEM_DEFINE_CLASS( jive::implict::EigensolveModule );


JIVE_BEGIN_PACKAGE( implict )


using jem::newInstance;
using jem::IllegalInputException;
using jem::io::endl;
using jem::io::space;
using jem::io::Writer;
using jem::io::PrintWriter;
using jive::util::joinNames;
using jive::util::XTable;
using jive::algebra::SparseMatrixObj;
using jive::algebra::SparseMatrixExt;
using jive::solver::Constrainer;
using jive::solver::EigenSolver;
using jive::model::Model;


//=======================================================================
//   class EigensolveModule::MatExpr_
//=======================================================================


class EigensolveModule::MatExpr_
{
 public:

  inline                     MatExpr_ ();

  void                       parseFrom

    ( const String&            expr );


 public:

  int                        size;
  int                        operands[3];
  double                     coeffs[3];

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline EigensolveModule::MatExpr_::MatExpr_ ()
{
  size = 0;

  for ( int i = 0; i < 3; i++ )
  {
    operands[i] = -1;
    coeffs[i]   =  0.0;
  }
}


//-----------------------------------------------------------------------
//   parseFrom
//-----------------------------------------------------------------------


void EigensolveModule::MatExpr_::parseFrom ( const String& expr )
{
  using jem::ParseException;
  using jem::util::Tokenizer;

  const char**    NAMES  = model::ActionParams::MATRIX;

  Ref<Tokenizer>  tokzer = newInstance<Tokenizer> ( expr );

  String          name;
  double          coeff;
  int             token;
  int             i, k;


  tokzer->setOptions ( Tokenizer::PARSE_SYMBOLS |
                       Tokenizer::PARSE_NUMBERS );

  k     = size = 0;
  token = tokzer->nextToken ();

  while ( token != Tokenizer::EOF_TOKEN )
  {
    if ( k >= 3 )
    {
      throw ParseException (
        JEM_FUNC,
        "too many matrix operands (maximum is three)"
      );
    }

    coeff = 1.0;

    if      ( token == '-' )
    {
      coeff = -1.0;
      token = tokzer->nextToken ();
    }
    else if ( token == '+' )
    {
      token = tokzer->nextToken ();
    }

    if ( token == Tokenizer::FLOAT_TOKEN ||
         token == Tokenizer::INTEGER_TOKEN )
    {
      coeff *= tokzer->getFloat ();

      if ( tokzer->nextToken() != '*' )
      {
        tokzer->parseError ( "expected a `*\'" );
      }

      token = tokzer->nextToken ();
    }

    if ( jem::isTiny( coeff ) )
    {
      throw ParseException (
        JEM_FUNC,
        "zero matrix coefficient"
      );
    }

    if ( token != Tokenizer::SYMBOL_TOKEN )
    {
      tokzer->parseError ( "expected a matrix name" );
    }

    name = tokzer->getSymbol ();

    for ( i = 0; i < 3; i++ )
    {
      if ( name == NAMES[i] )
      {
        break;
      }
    }

    if ( i >= 3 )
    {
      throw ParseException (
        JEM_FUNC,
        String::format (
          "invalid matrix: `%s\'; should be "
          "`%s\', `%s\' or `%s\'",
          name,
          NAMES[0], NAMES[1], NAMES[2]
        )
      );
    }

    coeffs[k]   = coeff;
    operands[k] = i;

    k++;

    token = tokzer->nextToken ();

    if ( token != '+' && token != '-' )
    {
      break;
    }
  }

  if ( token != Tokenizer::EOF_TOKEN )
  {
    tokzer->parseError ();
  }

  size = k;
}


//=======================================================================
//   class EigensolveModule::Utils_
//=======================================================================


class EigensolveModule::Utils_
{
 public:

  static inline void        configure

    ( PrintWriter&            prn );

  static void               printValues

    ( Writer&                 out,
      const Vector&           evals );

  static void               printValues

    ( Writer&                 out,
      const Matrix&           evals );

  static void               storeVectors

    ( XTable&                 table,
      const Matrix&           evecs,
      const DofSpace&         dofs,
      const Properties&       globdat );

};


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


inline void EigensolveModule::Utils_::configure ( PrintWriter& prn )
{
  using jem::io::NumberFormat;

  prn.nformat.setFlags          ( NumberFormat::SCIENTIFIC );
  prn.nformat.setFloatWidth     ( 13 );
  prn.nformat.setFractionDigits (  6 );
}


//-----------------------------------------------------------------------
//   printValues (real)
//-----------------------------------------------------------------------


void EigensolveModule::Utils_::printValues

  ( Writer&        out,
    const Vector&  evals )

{
  Ref<PrintWriter>  prn = newInstance<PrintWriter> ( &out );

  const idx_t  n = evals.size ();


  configure ( *prn );

  prn->incrIndentLevel ();

  print ( *prn, "\neigenvalues :" );

  if ( n <= 4 )
  {
    print ( *prn, space, evals, endl );
  }
  else
  {
    print ( *prn, "\n\n", evals[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      if ( i % 4 == 0 )
      {
        print ( *prn, ",\n" );
      }
      else
      {
        print ( *prn, ", " );
      }

      print ( *prn, evals[i] );
    }

    print ( *prn, endl );
  }

  prn->flush ();
}


//-----------------------------------------------------------------------
//   printValues (complex)
//-----------------------------------------------------------------------


void EigensolveModule::Utils_::printValues

  ( Writer&        out,
    const Matrix&  evals )

{
  Ref<PrintWriter>  prn = newInstance<PrintWriter> ( &out );

  const idx_t  n = evals.size (1);


  configure ( *prn );

  prn->incrIndentLevel ();

  print ( *prn, "\neigenvalues :" );

  if ( n <= 1 )
  {
    print ( *prn, space );
  }
  else
  {
    print ( *prn, "\n\n" );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( i > 0 )
    {
      if ( i % 2 == 0 )
      {
        print ( *prn, ",\n" );
      }
      else
      {
        print ( *prn, ", " );
      }
    }

    print ( *prn, '(', evals(0,i), ',', evals(1,i), ')' );
  }

  print ( *prn, endl );

  prn->flush ();
}


//-----------------------------------------------------------------------
//   storeVectors
//-----------------------------------------------------------------------


void EigensolveModule::Utils_::storeVectors

  ( XTable&            table,
    const Matrix&      evecs,
    const DofSpace&    dofs,
    const Properties&  globdat )

{
  StringVector  typeNames = dofs.getTypeNames  ();

  const idx_t   vecCount  = evecs    .size     (1);
  const idx_t   rowCount  = table    .rowCount ();
  const idx_t   typeCount = typeNames.size     ();

  IdxVector     idofs     ( rowCount );
  IdxVector     irows     ( rowCount );
  IdxVector     jcols     (        1 );

  Matrix        mbuf      ( rowCount, 1 );
  Vector        vbuf      ( mbuf[0] );

  idx_t         i, j, n;


  table.clearAll ();
  table.reserve  ( evecs.size(0) * evecs.size(1) );

  for ( j = 0; j < vecCount; j++ )
  {
    for ( i = 0; i < typeCount; i++ )
    {
      table.addColumn (
        String::format ( "v%d(%s)", j, typeNames[i] )
      );
    }
  }

  jcols[0] = 0;

  for ( j = 0; j < vecCount; j++ )
  {
    Vector  evec = evecs[j];

    for ( i = 0; i < typeCount; i++ )
    {
      n = dofs.getDofsForType ( idofs, irows, i );

      vbuf[slice(BEGIN,n)] = evec[idofs[slice(BEGIN,n)]];

      table.setBlock ( irows[slice(BEGIN,n)],
                       jcols,
                       mbuf(slice(BEGIN,n),ALL) );

      jcols[0]++;
    }
  }

  table.store ( table.getName(), globdat );
}


//=======================================================================
//   class EigensolveModule::RunData_
//=======================================================================


class EigensolveModule::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;


  enum                    { LEFT  = 0,
                            RIGHT = 1 };


  explicit inline           RunData_

    ( const String&           context );

  void                      init

    ( const String&           myName,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                      update

    ( const Properties&       globdat );

  inline Ref<XTable>        getRealVectors    ();
  inline Ref<XTable>        getImagVectors    ();
  inline void               deleteVectors     ();


 public:

  String                    myContext;

  Ref<DofSpace>             dofs;
  Ref<EigenSolver>          solver;

  double                    scale;
  bool                      symmetric;


 private:

  Ref<XTable>               newTable_

    ( const String&           name )             const;

  void                      initMatrices_

    ( Ref<AbstractMatrix>     mat[2],
      const String&           myName,
      const Properties&       myConf,
      const Properties&       myProps,
      const Properties&       globdat );

  void                      getMatrices_

    ( const MatExpr_&         expr,
      const Properties&       globdat );

  void                      evalExpression_

    ( SparseMatrixObj&        mat,
      const MatExpr_&         expr );

  Ref<SparseMatrixObj>      newMBuffer_

    ( const String&           name,
      const MatExpr_&         expr );

  static void               resize_

    ( SparseMatrixObj&        mat,
      idx_t                   size );


 private:

  Properties                params_;

  Ref<Model>                model_;
  Ref<Constraints>          cons_;

  Ref<Constrainer>          conman_  [2];
  MatExpr_                  mexpr_   [2];
  Ref<SparseMatrixObj>      mbuf_    [2];
  Ref<AbstractMatrix>       matrices_[3];

  Ref<XTable>               realVectors_;
  Ref<XTable>               imagVectors_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline EigensolveModule::RunData_::RunData_

  ( const String&  ctx ) :

    myContext ( ctx      ),
    params_   ( "params" )

{
  scale     = 1.0;
  symmetric = false;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::init

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::model::ActionParams;
  using jive::solver::newConstrainer;
  using jive::solver::EigenSolverParams;
  using jive::solver::EigenSolverFactory;

  Ref<AbstractMatrix>   mat[2];
  Ref<Constraints>      cons;
  Ref<VectorSpace>      vspace;

  Properties  myConf  = conf .makeProps ( myName );
  Properties  myProps = props.findProps ( myName );

  String      name;
  bool        applyCons;


  applyCons = true;

  myProps.find ( applyCons, PropNames::APPLY_CONS );
  myConf .set  ( PropNames::APPLY_CONS, applyCons );

  model_ = Model   ::get ( globdat, myContext );
  dofs   = DofSpace::get ( globdat, myContext );

  initMatrices_ ( mat, myName, myConf, myProps, globdat );

  if ( applyCons )
  {
    cons_ = Constraints::get ( dofs, globdat );

    params_.set ( ActionParams::CONSTRAINTS, cons_ );

    for ( int i = LEFT; i <= RIGHT; i++ )
    {
      if ( mat[i] )
      {
        name       = joinNames      ( myName, "constrainer" );
        conman_[i] = newConstrainer ( name, cons_, mat[i] );
        mat[i]     = conman_[i]->getOutputMatrix ();
      }
    }
  }

  solver::declareEigenSolvers ();

  name   = joinNames ( myName, PropNames::SOLVER );
  vspace = VectorSpace::get ( dofs, globdat );

  Properties  solverParams =

    EigenSolverParams::newInstance ( mat[LEFT], mat[RIGHT],
                                     vspace );

  solver = EigenSolverFactory::newInstance ( name,  conf,
                                             props, solverParams );
  solver->configure ( props );
  solver->getConfig ( conf  );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::update

  ( const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;


  if ( cons_ )
  {
    model_->takeAction ( Actions::GET_CONSTRAINTS, params_, globdat );
    cons_ ->compress   ();
  }

  if ( matrices_[0] )
  {
    Vector  fint ( dofs->dofCount() );

    fint = 0.0;

    params_.set ( ActionParams::INT_VECTOR, fint );
  }

  for ( int i = 0; i < 3; i++ )
  {
    if ( matrices_[i] )
    {
      params_.set ( ActionParams::MATRIX[i], matrices_[i] );

      model_->takeAction ( Actions::UPD_MATRIX[i], params_, globdat );
    }
  }

  if ( matrices_[0] )
  {
    params_.erase ( ActionParams::INT_VECTOR );
  }

  for ( int i = LEFT; i <= RIGHT; i++ )
  {
    if ( mbuf_[i] )
    {
      evalExpression_ ( *mbuf_[i], mexpr_[i] );
    }

    if ( conman_[i] )
    {
      conman_[i]->update ();
    }
  }
}


//-----------------------------------------------------------------------
//   getRealVectors
//-----------------------------------------------------------------------


inline Ref<XTable> EigensolveModule::RunData_::getRealVectors ()
{
  if ( ! realVectors_ )
  {
    realVectors_ = newTable_ ( "realEigenVectors" );
  }

  return realVectors_;
}


//-----------------------------------------------------------------------
//   getImagVectors
//-----------------------------------------------------------------------


inline Ref<XTable> EigensolveModule::RunData_::getImagVectors ()
{
  if ( ! imagVectors_ )
  {
    imagVectors_ = newTable_ ( "imagEigenVectors" );
  }

  return imagVectors_;
}


//-----------------------------------------------------------------------
//   deleteVectors
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::deleteVectors ()
{
  realVectors_ = nullptr;
  imagVectors_ = nullptr;
}


//-----------------------------------------------------------------------
//   newTable_
//-----------------------------------------------------------------------


Ref<XTable> EigensolveModule::RunData_::newTable_

  ( const String&  name ) const

{
  JEM_PRECHECK ( dofs );

  using jive::util::DenseTable;
  using jive::util::SparseTable;

  if ( dofs->isDense() )
  {
    return newInstance<DenseTable>  ( name, dofs->getItems() );
  }
  else
  {
    return newInstance<SparseTable> ( name, dofs->getItems() );
  }
}


//-----------------------------------------------------------------------
//   initMatrices_
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::initMatrices_

  ( Ref<AbstractMatrix>  mat[2],
    const String&        myName,
    const Properties&    myConf,
    const Properties&    myProps,
    const Properties&    globdat )

{
  using jem::util::None;

  const char*  PROPS[2] = { PropNames::LEFT_MATRIX,
                            PropNames::RIGHT_MATRIX };

  String       expr     = "matrix0";


  for ( int i = LEFT; i <= RIGHT; i++ )
  {
    myProps.find ( expr, PROPS[i] );

    try
    {
      mexpr_[i].parseFrom ( expr );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( myProps.getContext( PROPS[i] ) );
      throw;
    }

    myConf.set ( PROPS[i], expr );

    expr = "";
  }

  if ( mexpr_[LEFT].size == 0 )
  {
    myProps.propertyError ( PROPS[LEFT], "empty expression" );
  }

  getMatrices_ ( mexpr_[LEFT], globdat );

  if ( mexpr_[RIGHT].size > 0 )
  {
    getMatrices_ ( mexpr_[RIGHT], globdat );

    for ( int i = 0; i < mexpr_[LEFT].size; i++ )
    {
      mexpr_[LEFT].coeffs[i] /= mexpr_[RIGHT].coeffs[0];
    }

    for ( int i = 1; i < mexpr_[RIGHT].size; i++ )
    {
      mexpr_[RIGHT].coeffs[i] /= mexpr_[RIGHT].coeffs[0];
    }

    mexpr_[RIGHT].coeffs[0] = 1.0;
  }

  scale = mexpr_[LEFT].coeffs[0];

  mexpr_[LEFT].coeffs[0] = 1.0;

  for ( int i = 1; i < mexpr_[LEFT].size; i++ )
  {
    mexpr_[LEFT].coeffs[i] /= scale;
  }

  symmetric = true;

  for ( int i = 0; i < 3; i++ )
  {
    if ( matrices_[i] && ! matrices_[i]->isSymmetric() )
    {
      symmetric = false;
      break;
    }
  }

  for ( int i = LEFT; i <= RIGHT; i++ )
  {
    switch ( mexpr_[i].size )
    {
    case 0:

      mat[i] = nullptr;
      break;

    case 1:

      mat[i] = matrices_[mexpr_[i].operands[0]];
      break;

    default:

      mbuf_[i] = newMBuffer_ ( joinNames( myName, PROPS[i] ),
                               mexpr_[i] );
      mat[i]   = mbuf_[i];
    }
  }
}


//-----------------------------------------------------------------------
//   getMatrices_
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::getMatrices_

  ( const MatExpr_&    expr,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

  for ( int i = 0; i < expr.size; i++ )
  {
    int  j = expr.operands[i];

    if ( matrices_[j] )
    {
      continue;
    }

    model_->takeAction ( Actions::NEW_MATRIX[j],
                         params_, globdat );

    if ( ! params_.find( matrices_[j], ActionParams::MATRIX[j] ) )
    {
      String  name = model_->getName ();

      throw IllegalInputException (
        myContext,
        String::format (
          "failed to get %s (model `%s\' ignored the action `%s\')",
          ActionParams::MATRIX[j],
          name,
          Actions::NEW_MATRIX[j]
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   evalExpression_
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::evalExpression_

  ( SparseMatrixObj&  mat,
    const MatExpr_&   expr )

{
  using jive::model::ActionParams;

  const idx_t       dofCount = dofs->dofCount ();

  SparseMatrixExt*  ext;
  SparseMatrix      smt;


  if ( mat.size(0) != dofCount ||
       mat.size(1) != dofCount )
  {
    resize_ ( mat, dofCount );
  }
  else
  {
    mat.setToZero ();
  }

  for ( int i = 0; i < expr.size; i++ )
  {
    int  j = expr.operands[i];

    ext = matrices_[j]->getExtension<SparseMatrixExt> ();
    smt = ext->toSparseMatrix ();

    if ( mat.size(0) != smt.size(0) ||
         mat.size(1) != smt.size(1) )
    {
      throw IllegalInputException (
        myContext,
        String::format (
          "%s has wrong shape: (%d x %d); should be (%d x %d)",
          ActionParams::MATRIX[j],
          smt.size(0), smt.size(1),
          mat.size(0), mat.size(1)
        )
      );
    }

    mat.addMatrix ( smt, expr.coeffs[i] );
  }
}


//-----------------------------------------------------------------------
//   newMBuffer_
//-----------------------------------------------------------------------


Ref<SparseMatrixObj> EigensolveModule::RunData_::newMBuffer_

  ( const String&    name,
    const MatExpr_&  expr )

{
  using jive::model::ActionParams;

  SparseMatrixObj
    ::Traits      traits = SparseMatrixObj::SYMMETRIC;

  for ( int i = 0; i < expr.size; i++ )
  {
    int  j = expr.operands[i];

    if ( ! matrices_[j]->isSymmetric() )
    {
      traits = 0;
    }

    if ( ! matrices_[j]->hasExtension<SparseMatrixExt>() )
    {
      throw IllegalInputException (
        myContext,
        String::format (
          "%s does not implement the sparse matrix extension",
          ActionParams::MATRIX[j]
        )
      );
    }
  }

  return newInstance<SparseMatrixObj> ( name, traits );
}



//-----------------------------------------------------------------------
//   resize_
//-----------------------------------------------------------------------


void EigensolveModule::RunData_::resize_

  ( SparseMatrixObj&  mat,
    idx_t             size )

{
  IdxVector  offsets ( size + 1 );
  IdxVector  indices;

  offsets = 0_idx;

  mat.setMatrix (
    SparseMatrix ( jem::shape( size, size ), offsets, indices )
  );
}


//=======================================================================
//   class EigensolveModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  EigensolveModule::TYPE_NAME = "Eigensolve";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


EigensolveModule::EigensolveModule ( const String& name ) :

  Super ( name )

{
  eigenCounts_[0] = 0;
  eigenCounts_[1] = 0;
  printValues_    = false;
  storeVectors_   = false;
}


EigensolveModule::~EigensolveModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status EigensolveModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  rundat_ = nullptr;

  Ref<RunData_>  newdat = newInstance<RunData_> ( getContext() );

  newdat->init ( myName_, conf, props, globdat );
  rundat_.swap ( newdat );

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status EigensolveModule::run ( const Properties& globdat )
{
  using jem::min;
  using jem::sum;
  using jem::System;

  if ( ! rundat_ )
  {
    return DONE;
  }

  RunData_&    d        = * rundat_;

  const idx_t  dofCount = d.dofs->dofCount ();

  idx_t        m, n;


  n = min ( dofCount, (eigenCounts_[0] + eigenCounts_[1]) );
  m = min ( n,         eigenCounts_[0] );

  print ( System::info ( myName_ ), "Updating matrices ...\n" );

  d.update ( globdat );

  if ( ! d.symmetric )
  {
    Matrix  evals ( 2, n );

    if ( ! storeVectors_ )
    {
      print ( System::info ( myName_ ),
              "Computing complex eigenvalues ...\n" );

      d.solver->getEigenValues ( evals, m );
    }
    else
    {
      Ref<XTable>  realVecs = d.getRealVectors ();
      Ref<XTable>  imagVecs = d.getImagVectors ();

      Cubix        evecs    ( 2, dofCount, n );

      print ( System::info ( myName_ ),
              "Computing complex "
              "eigenvalues and vectors ...\n" );

      d.solver->getEigenVectors ( evals, evecs, m );

      Utils_::storeVectors ( *realVecs, evecs(0,ALL,ALL),
                             *d.dofs,   globdat );
      Utils_::storeVectors ( *imagVecs, evecs(1,ALL,ALL),
                             *d.dofs,   globdat );
    }

    evals *= d.scale;

    if ( printValues_ )
    {
      Utils_::printValues ( System::info( myName_ ), evals );
    }

    globdat.set ( Globdat::REAL_EIGENVALS, evals(0,ALL) );
    globdat.set ( Globdat::IMAG_EIGENVALS, evals(1,ALL) );
  }
  else
  {
    Vector  evals ( n );

    if ( ! storeVectors_ )
    {
      print ( System::info ( myName_ ),
              "Computing real eigenvalues ...\n" );

      d.solver->getRealEigenValues ( evals, m );
    }
    else
    {
      Ref<XTable>  realVecs = d.getRealVectors ();

      Matrix       evecs    ( dofCount, n );

      print ( System::info ( myName_ ),
              "Computing real eigenvalues and vectors ...\n" );

      d.solver->getRealEigenVectors ( evals, evecs, m );

      Utils_::storeVectors ( *realVecs, evecs,
                             *d.dofs,   globdat );
    }

    evals *= d.scale;

    if ( printValues_ )
    {
      Utils_::printValues ( System::info( myName_ ), evals );
    }

    globdat.set ( Globdat::REAL_EIGENVALS, evals );
  }

  print ( System::info( myName_ ), endl );

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void EigensolveModule::shutdown ( const Properties& globdat )
{
  rundat_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void EigensolveModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    IntVector   eigenCounts ( 2 );

    myProps.find ( printValues_,  PropNames::PRINT_VALUES  );
    myProps.find ( storeVectors_, PropNames::STORE_VECTORS );

    if ( ! storeVectors_ && rundat_ )
    {
      rundat_->deleteVectors ();
    }

    if ( myProps.find( eigenCounts, PropNames::EIGEN_COUNTS  ) )
    {
      if ( eigenCounts.size() != 2 )
      {
        myProps.propertyError (
          PropNames::EIGEN_COUNTS,
          "invalid array length; should be two"
        );
      }

      eigenCounts_[0] = eigenCounts[0];
      eigenCounts_[1] = eigenCounts[1];
    }
  }

  if ( rundat_ )
  {
    rundat_->solver->configure ( props );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void EigensolveModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  IntVector   eigenCounts ( 2 );

  eigenCounts[0] = (int) eigenCounts_[0];
  eigenCounts[1] = (int) eigenCounts_[1];

  myConf.set ( PropNames::EIGEN_COUNTS,  eigenCounts   );
  myConf.set ( PropNames::PRINT_VALUES,  printValues_  );
  myConf.set ( PropNames::STORE_VECTORS, storeVectors_ );

  if ( rundat_ )
  {
    rundat_->solver->getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> EigensolveModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void EigensolveModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( implict )
