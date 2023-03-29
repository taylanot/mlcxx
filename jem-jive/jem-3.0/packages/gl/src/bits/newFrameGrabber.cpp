
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/base/IllegalInputException.h>
#include <jem/io/utilities.h>
#include <jem/io/FileName.h>
#include <jem/gl/PPMGrabber.h>
#include <jem/gl/PNGGrabber.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class FrameGrabber
//=======================================================================

//-----------------------------------------------------------------------
//   byType
//-----------------------------------------------------------------------


Ref<FrameGrabber>  FrameGrabber::byType ( const String& type )
{
  using jem::io::canZip;

  if      ( type.equalsIgnoreCase( "png" ) && canZip() )
  {
    return newInstance<PNGGrabber> ();
  }
  else if ( type.equalsIgnoreCase( "ppm" ) )
  {
    return newInstance<PPMGrabber> ();
  }
  else
  {
    String  msg =

      String::format ( "unsupported image type: `%s\'; ", type );

    if ( canZip() )
    {
      msg = msg + "supported types are `png\' and `ppm\'";
    }
    else
    {
      msg = msg + "supported type is `ppm\'";
    }

    throw IllegalInputException ( JEM_FUNC, msg );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   byFile
//-----------------------------------------------------------------------


Ref<FrameGrabber> FrameGrabber::byFile ( const String& fname )
{
  idx_t  i = io::FileName::findSuffix ( fname );

  if ( i < 0 || i == fname.size() - 1 )
  {
    throw IllegalInputException (
      JEM_FUNC,
      String::format ( "unknown image file type: `%s\'", fname )
    );
  }

  return byType ( fname[slice(i + 1,END)] );
}


JEM_END_PACKAGE( gl )
