
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


#include <jem/util/Properties.h>
#include <jive/mesh/Names.h>
#include <jive/mesh/ElementTypes.h>


JIVE_BEGIN_PACKAGE( mesh )


//=======================================================================
//   class ElementTypes
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ElementTypes::QUAD4     = "Quad4";
const char*  ElementTypes::QUAD8     = "Quad8";
const char*  ElementTypes::QUAD12    = "Quad12";
const char*  ElementTypes::TRIANGLE3 = "Triangle3";
const char*  ElementTypes::TRIANGLE6 = "Triangle6";


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


bool ElementTypes::configure

  ( int&               type,
    const Properties&  props,
    const String&      choice0,
    const String&      choice1 )

{
  String  typeName;

  if ( props.find( typeName, PropNames::ELEM_TYPE ) )
  {
    if      ( typeName == choice0 )
    {
      type = 0;
    }
    else if ( typeName == choice1 )
    {
      type = 1;
    }
    else
    {
      props.propertyError (
        PropNames::ELEM_TYPE,
        String::format (
          "invalid element type: %s; valid types are "
          "`%s\' and `%s\'",
          typeName,
          choice0,
          choice1
        )
      );
    }

    return true;
  }

  return false;
}


bool ElementTypes::configure

  ( int&               type,
    const Properties&  props,
    const String&      choice0,
    const String&      choice1,
    const String&      choice2 )

{
  String  typeName;

  if ( props.find( typeName, PropNames::ELEM_TYPE ) )
  {
    if      ( typeName == choice0 )
    {
      type = 0;
    }
    else if ( typeName == choice1 )
    {
      type = 1;
    }
    else if ( typeName == choice2 )
    {
      type = 2;
    }
    else
    {
      props.propertyError (
        PropNames::ELEM_TYPE,
        String::format (
          "invalid element type: %s; valid types are "
          "`%s\', `%s\' and `%s\'",
          typeName,
          choice0,
          choice1,
          choice2
        )
      );
    }

    return true;
  }

  return false;
}


JIVE_END_PACKAGE( mesh )
