
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
#include "ConstantCond.h"


JEM_DEFINE_SERIABLE_CLASS( jive::util::ConstantCond );


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class ConstantCond
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


ConstantCond::~ConstantCond ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ConstantCond::readFrom ( ObjectInput& in )
{
  decode ( in, value );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ConstantCond::writeTo ( ObjectOutput& out ) const
{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String ConstantCond::toString () const
{
  return String ( value );
}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


idx_t ConstantCond::argCount () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


double ConstantCond::getValue ( const double* args ) const
{
  return (value ? 1.0 : 0.0);
}


//-----------------------------------------------------------------------
//   getDeriv
//-----------------------------------------------------------------------


double ConstantCond::getDeriv

  ( idx_t          iarg,
    const double*  args ) const

{
  return 0.0;
}


//-----------------------------------------------------------------------
//   markUsedArgs
//-----------------------------------------------------------------------


idx_t ConstantCond::markUsedArgs ( bool* mask ) const
{
  return 0;
}


JIVE_END_PACKAGE( util )
