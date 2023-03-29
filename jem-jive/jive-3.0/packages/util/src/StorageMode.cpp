
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
#include <jive/util/StorageMode.h>


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   data
//=======================================================================


const int    STORAGE_MODE_COUNT = 3;

const char*  STORAGE_MODES[]    = { "Default",
                                    "Compact",
                                    "Minimal" };

const char*  STORAGE_MODE_PROP  = "storageMode";


//=======================================================================
//   functions
//=======================================================================

//-----------------------------------------------------------------------
//   getStorageMode
//-----------------------------------------------------------------------


StorageMode getStorageMode ( const String& mstr )
{
  StorageMode  mode = DEFAULT_STORAGE;

  getStorageMode ( mode, mstr );

  return mode;
}


bool                  getStorageMode

  ( StorageMode&        mode,
    const String&       mstr )

{
  const char**  MODES = STORAGE_MODES;

  if      ( mstr == MODES[DEFAULT_STORAGE] )
  {
    mode = DEFAULT_STORAGE;
  }
  else if ( mstr == MODES[COMPACT_STORAGE] )
  {
    mode = COMPACT_STORAGE;
  }
  else if ( mstr == MODES[MINIMAL_STORAGE] )
  {
    mode = MINIMAL_STORAGE;
  }
  else
  {
    return false;
  }

  return true;
}


bool                  getStorageMode

  ( StorageMode&        mode,
    const Properties&   conf,
    const Properties&   props )

{
  String  mstr  = STORAGE_MODES[mode];
  bool    found = props.find ( mstr, STORAGE_MODE_PROP );

  if ( found )
  {
    if ( ! getStorageMode( mode, mstr ) )
    {
      const char**  MODES = STORAGE_MODES;

      props.propertyError (
        STORAGE_MODE_PROP,
        String::format (
          "invalid storage mode: `%s\';\n"
          "valid choices are   : `%s\', `%s\' and `%s\'",
          mstr,
          MODES[DEFAULT_STORAGE],
          MODES[COMPACT_STORAGE],
          MODES[MINIMAL_STORAGE]
        )
      );
    }
  }

  conf.set ( STORAGE_MODE_PROP, mstr );

  return found;
}


JIVE_END_PACKAGE( util )
