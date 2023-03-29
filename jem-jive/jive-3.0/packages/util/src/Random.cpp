
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/Globdat.h>
#include <jive/util/Random.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::Random );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;


//=======================================================================
//   class Random
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Random::~Random ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Random::toString () const
{
  return String ( const_cast<Self*>( this ) -> rand_.next() );
}


//-----------------------------------------------------------------------
//   toInteger
//-----------------------------------------------------------------------


lint Random::toInteger () const
{
  return (lint) const_cast<Self*>( this ) ->
    rand_.next ( jem::maxOf<idx_t>() );
}


//-----------------------------------------------------------------------
//   toFloat
//-----------------------------------------------------------------------


double Random::toFloat () const
{
  return const_cast<Self*>( this ) -> rand_.next ();
}


//-----------------------------------------------------------------------
//   isFloat
//-----------------------------------------------------------------------


bool Random::isFloat () const
{
  return true;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void Random::readFrom ( ObjectInput& in )
{
  decode ( in, rand_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void Random::writeTo ( ObjectOutput& out ) const
{
  encode ( out, rand_ );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Random> Random::get ( const Properties& globdat )
{
  Ref<Random>  rand;

  if ( ! globdat.find( rand, Globdat::RANDOM ) )
  {
    rand = newInstance<Random> ();

    globdat.set ( Globdat::RANDOM, rand );
  }

  return rand;
}


//-----------------------------------------------------------------------
//   install
//-----------------------------------------------------------------------


void Random::install ( const Properties& globdat )
{
  globdat.set ( Globdat::RANDOM, newInstance<Random>() );
}


JIVE_END_PACKAGE( util )
