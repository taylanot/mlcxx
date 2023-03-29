
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/algebra/ConstrainedMatrix.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/PreconFactory.h>
#include <jive/solver/DualPrecon.h>


JEM_DEFINE_CLASS( jive::solver::DualPrecon );


JIVE_BEGIN_PACKAGE( solver )


using jem::newInstance;
using jem::dynamicCast;


//=======================================================================
//   class DualPrecon
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DualPrecon::TYPE_NAME     = "Dual";

const char*  DualPrecon::OPERATIONS[3] =
{
  "Add",
  "Multiply",
  "Multiply2"
};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DualPrecon::DualPrecon

  ( const String&        name,
    Ref<AbstractMatrix>  matrix,
    Ref<AbstractMatrix>  precon0,
    Ref<AbstractMatrix>  precon1,
    Ref<Constraints>     cons ) :

    Super   ( name   ),
    matrix_ ( matrix )

{
  JEM_PRECHECK ( matrix  &&
                 precon0 &&
                 precon1 );

  if ( cons )
  {
    Ref<ConstrainedMatrix>  conmat =

      jem::dynamicCast<ConstrainedMatrix> ( matrix );

    if ( conmat == nullptr || cons != conmat->getConstraints() )
    {
      matrix_ = newInstance<ConstrainedMatrix> ( matrix_, cons );
    }
  }

  precon_[0] = precon0;
  precon_[1] = precon1;

  if ( matrix_->isSymmetric() )
  {
    operation_ = MULTIPLY2;
  }
  else
  {
    operation_ = MULTIPLY;
  }

  connect_ ();
}


DualPrecon::~DualPrecon ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void DualPrecon::resetEvents ()
{
  matrix_->resetEvents ();
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


AbstractMatrix::Shape DualPrecon::shape () const
{
  return matrix_->shape ();
}


//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


void DualPrecon::start ()
{
  for ( int i = 0; i < 2; i++ )
  {
    Precon*  p = dynamicCast<Precon*> ( precon_[i] );

    if ( p )
    {
      p->start ();
    }
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void DualPrecon::finish ()
{
  for ( int i = 0; i < 2; i++ )
  {
    Precon*  p = dynamicCast<Precon*> ( precon_[i] );

    if ( p )
    {
      p->finish ();
    }
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void DualPrecon::update ()
{
  for ( int i = 0; i < 2; i++ )
  {
    Precon*  p = dynamicCast<Precon*> ( precon_[i] );

    if ( p )
    {
      p->update ();
    }
  }
}


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


void DualPrecon::matmul

  ( const Vector&  lhs,
    const Vector&  rhs ) const

{
  using jem::numeric::axpy;

  JEM_PRECHECK ( lhs.size() == rhs.size() );

  precon_[0]->matmul ( lhs, rhs );

  if ( operation_ == ADD )
  {
    Vector  tmp ( lhs.size() );

    precon_[1]->matmul ( tmp, rhs );

    lhs += tmp;
  }
  else
  {
    Matrix  tmp ( lhs.size(), 2 );

    Vector  du  ( tmp[0] );
    Vector  r   ( tmp[1] );


    matrix_->matmul ( r, lhs );

    axpy ( r, rhs, -1.0, r );

    precon_[1]->matmul ( du, r );

    lhs += du;

    if ( operation_ == MULTIPLY2 )
    {
      matrix_->matmul ( r, lhs );

      axpy ( r, rhs, -1.0, r );

      precon_[0]->matmul ( du, r );

      lhs += du;
    }
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DualPrecon::configure ( const Properties& props )
{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      op;

    if ( myProps.find( op, PropNames::OPERATION ) )
    {
      int  i;

      for ( i = 0; i < 3; i++ )
      {
        if ( op.equalsIgnoreCase( OPERATIONS[i] ) )
        {
          break;
        }
      }

      if ( i >= 3 )
      {
        myProps.propertyError (
          PropNames::OPERATION,
          String::format (
            "invalid operation: %s; "
            "should be one of `%s\', `%s\' or `%s\'",
            op,
            OPERATIONS[0],
            OPERATIONS[1],
            OPERATIONS[2]
          )
        );
      }

      operation_ = (Operation) i;

      newValuesEvent.emit ( *this );
    }
  }

  Super::configure ( props );

  for ( int i = 0; i < 2; i++ )
  {
    Precon*  p = dynamicCast<Precon*> ( precon_[i] );

    if ( p )
    {
      p->configure ( props );
    }
  }

  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DualPrecon::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::OPERATION, OPERATIONS[operation_] );

  Super::getConfig ( props );

  for ( int i = 0; i < 2; i++ )
  {
    Precon*  p = dynamicCast<Precon*> ( precon_[i] );

    if ( p )
    {
      p->getConfig ( props );
    }
  }
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool DualPrecon::hasTrait ( const String& trait ) const
{
  using jive::algebra::MatrixTraits;

  if      ( trait == MatrixTraits::SYMMETRIC )
  {
    return ( operation_ != MULTIPLY         &&
             matrix_   ->hasTrait ( trait ) &&
             precon_[0]->hasTrait ( trait ) &&
             precon_[1]->hasTrait ( trait ) );
  }
  else if ( trait == MatrixTraits::DISTRIBUTED )
  {
    return matrix_->hasTrait ( trait );
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


Constraints* DualPrecon::getConstraints () const
{
  Constraints*  cons = 0;

  for ( int i = 0; i < 2 && cons == 0; i++ )
  {
    Precon*  p = dynamicCast<Precon*> ( precon_[i] );

    if ( p )
    {
      cons = p->getConstraints ();
    }
  }

  return cons;
}


//-----------------------------------------------------------------------
//   getOperation
//-----------------------------------------------------------------------


DualPrecon::Operation DualPrecon::getOperation () const
{
  return operation_;
}


//-----------------------------------------------------------------------
//   setOperation
//-----------------------------------------------------------------------


void DualPrecon::setOperation ( Operation op )
{
  if ( op != operation_ )
  {
    operation_ = op;

    newValuesEvent.emit ( *this );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Precon> DualPrecon::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<AbstractMatrix>  matrix;
  Ref<Constraints>     cons;
  Ref<AbstractMatrix>  precon[2];


  if ( ! params.find( matrix, SolverParams::MATRIX ) )
  {
    return nullptr;
  }

  if ( params.find( cons, SolverParams::CONSTRAINTS ) )
  {
    Ref<ConstrainedMatrix>  conmat;

    params.find ( conmat, SolverParams::CONSTRAINED_MATRIX );

    if ( conmat == nullptr || cons != conmat->getConstraints() )
    {
      conmat = newInstance<ConstrainedMatrix> ( matrix, cons );

      params.set ( SolverParams::CONSTRAINED_MATRIX, conmat );
    }

    matrix = conmat;
  }

  for ( int i = 0; i < 2; i++ )
  {
    String  preconName =

      joinNames ( name, PropNames::PRECONDITIONER + String( i ) );

    precon[i] = newPrecon ( preconName,
                            conf, props, params, globdat );

    if ( ! precon[i] )
    {
      return nullptr;
    }
  }

  return newInstance<Self> ( name,      matrix,
                             precon[0], precon[1], cons );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DualPrecon::declare ()
{
  PreconFactory::declare ( TYPE_NAME,  & Self::makeNew );
  PreconFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void DualPrecon::connect_ ()
{
  using jem::util::connect;

  connect ( matrix_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( matrix_->newStructEvent, this, & Self::structChanged_ );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void DualPrecon::valuesChanged_ ()
{
  newValuesEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void DualPrecon::structChanged_ ()
{
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
