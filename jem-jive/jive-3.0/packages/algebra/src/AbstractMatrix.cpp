
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
#include <jem/io/PrintWriter.h>
#include <jive/algebra/AbstractMatrix.h>


JEM_DEFINE_CLASS( jive::algebra::AbstractMatrix );


JIVE_BEGIN_PACKAGE( algebra )


//=======================================================================
//   class MatrixTraits
//=======================================================================


const char*  MatrixTraits::SYMMETRIC   = "SYMMETRIC";
const char*  MatrixTraits::DISTRIBUTED = "DISTRIBUTED";


//=======================================================================
//   class AbstractMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


AbstractMatrix::AbstractMatrix ( const String& name ) :

  Super ( name )

{}


AbstractMatrix::~AbstractMatrix ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void AbstractMatrix::resetEvents ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String AbstractMatrix::getContext () const
{
  return NamedObject::makeContext ( "matrix", myName_ );
}


//-----------------------------------------------------------------------
//   printTo
//-----------------------------------------------------------------------


void AbstractMatrix::printTo ( PrintWriter& out ) const
{
  print ( out, toString() );
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* AbstractMatrix::getExtByID ( ExtensionID extID ) const
{
  return 0;
}


//-----------------------------------------------------------------------
//   hasTrait
//-----------------------------------------------------------------------


bool AbstractMatrix::hasTrait ( const String& trait ) const
{
  return false;
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void AbstractMatrix::emitEvents_ ( int events )
{}


JIVE_END_PACKAGE( algebra )
