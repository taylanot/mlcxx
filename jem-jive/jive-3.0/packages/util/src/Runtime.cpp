
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/Globdat.h>
#include <jive/util/Runtime.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::Runtime );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;


//=======================================================================
//   class Runtime
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Runtime::Runtime ()
{
  timer_.start ();
}


Runtime::~Runtime ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Runtime::toString () const
{
  return String ( timer_.toDouble() );
}


//-----------------------------------------------------------------------
//   toInteger
//-----------------------------------------------------------------------


lint Runtime::toInteger () const
{
  return timer_.time().sec ();
}


//-----------------------------------------------------------------------
//   toFloat
//-----------------------------------------------------------------------


double Runtime::toFloat () const
{
  return timer_.toDouble ();
}


//-----------------------------------------------------------------------
//   isFloat
//-----------------------------------------------------------------------


bool Runtime::isFloat () const
{
  return true;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void Runtime::readFrom ( ObjectInput& in )
{
  decode ( in, timer_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void Runtime::writeTo ( ObjectOutput& out ) const
{
  encode ( out, timer_ );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Runtime> Runtime::get ( const Properties& globdat )
{
  Ref<Runtime>  rt;

  if ( ! globdat.find( rt, Globdat::RUNTIME ) )
  {
    rt = newInstance<Runtime> ();

    globdat.set ( Globdat::RUNTIME, rt );
  }

  return rt;
}


//-----------------------------------------------------------------------
//   install
//-----------------------------------------------------------------------


void Runtime::install ( const Properties& globdat )
{
  globdat.set ( Globdat::RUNTIME, newInstance<Runtime>() );
}


JIVE_END_PACKAGE( util )
