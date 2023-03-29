
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
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ArithmeticException.h>
#include <jem/base/array/operators.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/algebra/Names.h>
#include <jive/algebra/GramSchmidtFactory.h>
#include <jive/algebra/ClassicalGramSchmidt.h>
#include <jive/algebra/IteratedGramSchmidt.h>


JEM_DEFINE_CLASS( jive::algebra::IteratedGramSchmidt );


using jem::newInstance;


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   class IteratedGramSchmidt
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  IteratedGramSchmidt::TYPE_NAME = "IGS";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


IteratedGramSchmidt::IteratedGramSchmidt

  ( const String&     name,
    Ref<GramSchmidt>  gs ) :

    gs_      ( gs ),
    maxIter_ ( 10 )

{
  JEM_PRECHECK2 ( gs, "NULL GramSchmidt object" );
}


IteratedGramSchmidt::~IteratedGramSchmidt ()
{}


//-----------------------------------------------------------------------
//   orthogonalize
//-----------------------------------------------------------------------


void IteratedGramSchmidt::orthogonalize

  ( const Vector& a,
    const Vector& x,
    const Matrix& y ) const

{
  using jem::Float;
  using jem::ArithmeticException;


  JEM_PRECHECK2 ( a.size() >= y.size(1),
                  "coefficient array is too small" );

  static const double  ALPHA = 0.5;

  const idx_t  n = y.size(1);

  Vector       b;
  double       q;
  idx_t        i;


  if ( scratch_.size() < (n + 2) )
  {
    idx_t  len = n + 2 + scratch_.size() / 4;

    const_cast<Self*>( this )->scratch_.resize ( len );
  }

  b.ref ( scratch_[slice(BEGIN,n + 2)] );

  q = 0.0;
  a = 0.0;

  for ( i = 0; i < maxIter_; i++ )
  {
    gs_->orthogonalize ( b, x, y );

    if ( i == 0 )
    {
      q = b[n];
    }

    a[slice(BEGIN,n)] += b[slice(BEGIN,n)];

    if ( Float::isNaN( b[n] ) || Float::isNaN( b[n + 1] ) )
    {
      throw ArithmeticException ( getContext (),
                                  "invalid vector norm: NaN" );
    }

    if ( b[n + 1] <= Float::EPSILON * q )
    {
      b[n + 1] = 0.0;
      break;
    }

    if ( b[n + 1] >= ALPHA * b[n] )
    {
      break;
    }
  }

  if ( i >= maxIter_ )
  {
    throw ArithmeticException (
      getContext (),
      String::format (
        "iterated Gram Schmidt failed to converge "
        "in %d iterations",
        maxIter_
      )
    );
  }

  if ( a.size() > n )
  {
    a[n] = q;
  }
  if ( a.size() > n + 1 )
  {
    a[n + 1] = b[n + 1];
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void IteratedGramSchmidt::configure ( const Properties& props )
{
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( maxIter_, PropNames::MAX_ITER,
                   1,        maxOf( maxIter_ ) );
  }

  gs_->configure ( props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void IteratedGramSchmidt::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::MAX_ITER, maxIter_ );

  gs_->getConfig ( props );
}


//-----------------------------------------------------------------------
//   setMaxIterCount
//-----------------------------------------------------------------------


void IteratedGramSchmidt::setMaxIterCount ( idx_t n )
{
  JEM_PRECHECK2 ( n > 0, "invalid maximum number of iterations" );

  maxIter_ = n;
}


//-----------------------------------------------------------------------
//   getMaxIterCount
//-----------------------------------------------------------------------


idx_t IteratedGramSchmidt::getMaxIterCount () const
{
  return maxIter_;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<GramSchmidt> IteratedGramSchmidt::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    Ref<VectorSpace>   vspace )

{
  using jive::util::joinNames;


  String            gsName  =

    joinNames ( myName, PropNames::GRAM_SCHMIDT );

  Properties        myProps = props.findProps ( myName );

  Ref<GramSchmidt>  gs;


  if ( myProps.contains( PropNames::GRAM_SCHMIDT ) )
  {
    gs = GSFactory::newInstance ( gsName, conf, props, vspace );
  }
  else
  {
    Properties  gsConf = conf.makeProps ( gsName );

    gsConf.set ( "type", ClassicalGS::TYPE_NAME );

    gs = newInstance<ClassicalGS> ( gsName, vspace );
  }

  return newInstance<Self> ( myName, gs );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void IteratedGramSchmidt::declare ()
{
  GSFactory::declare ( TYPE_NAME,  & makeNew );
  GSFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( algebra )
