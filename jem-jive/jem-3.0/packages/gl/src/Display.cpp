
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


#include <cstring>
#include <jem/base/SerialSection.h>
#include <jem/gl/DisplayException.h>
#include <jem/gl/Display.h>
#include <jem/gl/Font.h>
#include "NativeViewer.h"
#include "NativeDisplay.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Display
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Display::NEW_THREAD    = 1 << 0;
const int  Display::OFF_SCREEN    = 1 << 1;
const int  Display::EXIT_ON_CLOSE = 1 << 2;


//-----------------------------------------------------------------------
//   open
//-----------------------------------------------------------------------


int Display::open ( int mode )
{
  char   progname[32];

  char*  argv[2] = { progname, nullptr };
  int    argc    = 1;

  ::strcpy ( progname, "<unknown program>" );

  return open ( argc, argv, mode );
}


int Display::open

  ( int&    argc,
    char**  argv,
    int     mode )

{
  SerialSection   ss;
  NativeDisplay*  disp;

  disp = NativeDisplay::getInstance ();

  if ( ! disp )
  {
    int  dpi = 0;

    if ( mode & OFF_SCREEN )
    {
      disp = newOffScreenDisplay ();
    }
    else
    {
      disp = newOnScreenDisplay ();
    }

    disp->init ( argc, argv, mode );

    dpi = disp->getScreenDPI ();

    if ( dpi > 0 )
    {
      if      ( dpi < 75 )
      {
        Font::setDefault ( PROPORTIONAL_SMALL_FONT );
      }
      else if ( dpi < 125 )
      {
        Font::setDefault ( PROPORTIONAL_MEDIUM_FONT );
      }
      else
      {
        Font::setDefault ( PROPORTIONAL_LARGE_FONT );
      }
    }
  }

  return disp->getMode ();
}


//-----------------------------------------------------------------------
//   isOpen
//-----------------------------------------------------------------------


bool Display::isOpen ()
{
  bool  retval = false;

  {
    SerialSection  ss;

    retval = (NativeDisplay::getInstance() != 0);
  }

  return retval;
}


//-----------------------------------------------------------------------
//   hasWindows
//-----------------------------------------------------------------------


bool Display::hasWindows ()
{
  bool  retval = false;

  {
    SerialSection  ss;

    if ( NativeDisplay::getInstance() )
    {
      retval = NativeDisplay::getInstance()->hasWindows ();
    }
  }

  return retval;
}


//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


int Display::getMode ()
{
  int  retval = 0;

  {
    SerialSection   ss;

    NativeDisplay*  disp = NativeDisplay::getInstance ();

    if ( disp )
    {
      retval = disp->getMode ();
    }
  }

  return retval;
}


//-----------------------------------------------------------------------
//   mainLoop
//-----------------------------------------------------------------------


void Display::mainLoop ()
{
  if ( ! isOpen() )
  {
    closedError_ ();
  }

  NativeDisplay::getInstance()->mainLoop ();
}


//-----------------------------------------------------------------------
//   newWindow
//-----------------------------------------------------------------------


Ref<Viewer> Display::newWindow

  ( const String&      title,
    int                width,
    int                height,
    Ref<ErrorHandler>  handler )

{
  if ( ! isOpen() )
  {
    closedError_ ();
  }

  Ref<NativeWindow> win =

    NativeDisplay::getInstance()->newWindow ( title,
                                              width, height,
                                              handler );

  return newInstance<NativeViewer> ( win );
}


//-----------------------------------------------------------------------
//   getThread
//-----------------------------------------------------------------------


Thread* Display::getThread ()
{
  Thread*  t = 0;

  {
    SerialSection  ss;

    t = NativeDisplay::getThread ();
  }

  return t;
}


//-----------------------------------------------------------------------
//   closedError_
//-----------------------------------------------------------------------


void Display::closedError_ ()
{
  throw DisplayException (
    "jem::gl::Display",
    "The display has not yet been opened; "
    "call Display::open first"
  );
}


JEM_END_PACKAGE( gl )
