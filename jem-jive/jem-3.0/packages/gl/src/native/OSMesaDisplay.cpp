
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


#include <jem/base/assert.h>
#include <jem/base/Thread.h>
#include <jem/base/SystemException.h>
#include <jem/io/StringReader.h>
#include "NativeDisplay.h"

extern "C"
{
  #include <GL/osmesa.h>
}


#undef  OSMESA_VERSION
#define OSMESA_VERSION (OSMESA_MAJOR_VERSION * 100 + \
                        OSMESA_MINOR_VERSION)


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class OSMesaWindow
//=======================================================================


class OSMesaWindow : public NativeWindow
{
 public:

  typedef OSMesaWindow    Self;
  typedef NativeWindow    Super;


                          OSMesaWindow

    ( int                   width,
      int                   height );

  virtual void            close           ()       override;
  virtual void            setCurrent      ()       override;
  virtual void            swapBuffers     ()       override;
  virtual Vertex2i        getMousePos     () const override;
  virtual Size2i          getWindowSize   () const override;
  inline bool             isOpen          () const override;


 protected:

  virtual                ~OSMesaWindow    ();


 private:

  void                    close_          ();
  GLboolean               setCurrent_     ();


 private:

  OSMesaContext           context_;
  GLenum                  ctype_;
  byte*                   buffer_;
  lint                    bufsize_;
  int                     width_;
  int                     height_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


OSMesaWindow::OSMesaWindow

  ( int  width,
    int  height )

{
  int  maxWidth;
  int  maxHeight;


#if (OSMESA_VERSION >= 400)

  OSMesaGetIntegerv ( OSMESA_MAX_WIDTH,  &maxWidth );
  OSMesaGetIntegerv ( OSMESA_MAX_HEIGHT, &maxHeight );

#else

  maxWidth = maxHeight = 4 * 1024;

#endif

  if      ( width < 0 )
  {
    width = 200;
  }
  else if ( width < 1 )
  {
    width = 1;
  }
  else if ( width > maxWidth )
  {
    width = maxWidth;
  }

  if      ( height < 0 )
  {
    height = 200;
  }
  else if ( height < 1 )
  {
    height = 1;
  }
  else if ( height > maxHeight )
  {
    height = maxHeight;
  }

  context_ = OSMesaCreateContext ( OSMESA_RGBA, nullptr );

  if ( context_ == 0 )
  {
    throw SystemException (
      JEM_FUNC,
      "failed to create a new off-screen rendering context "
      "(error calling OSMesaCreateContext)"
    );
  }

  ctype_   = GL_NONE;
  buffer_  = nullptr;
  bufsize_ = 0_lint;
  width_   = width;
  height_  = height;
}


OSMesaWindow::~OSMesaWindow ()
{
  close_ ();
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void OSMesaWindow::close ()
{
  Super::close  ();
  this-> close_ ();
}


//-----------------------------------------------------------------------
//   setCurrent
//-----------------------------------------------------------------------


void OSMesaWindow::setCurrent ()
{
  JEM_ASSERT ( isOpen() );

  if ( OSMesaGetCurrentContext() == context_ )
  {
    return;
  }

  GLboolean  result;

  if ( ! buffer_ )
  {
    result = setCurrent_ ();
  }
  else
  {
    result = OSMesaMakeCurrent ( context_, buffer_,
                                 ctype_,   width_, height_ );
  }

  if ( result != GL_TRUE )
  {
    throw SystemException (
      JEM_FUNC,
      "failed to set the current off-screen rendering context "
      "(error calling OSMesaMakeCurrent)"
    );
  }
}

//-----------------------------------------------------------------------
//   swapBuffers
//-----------------------------------------------------------------------


void OSMesaWindow::swapBuffers ()
{}


//-----------------------------------------------------------------------
//   getMousePos
//-----------------------------------------------------------------------


Vertex2i OSMesaWindow::getMousePos () const
{
  return Vertex2i ( 0, 0 );
}


//-----------------------------------------------------------------------
//   getWindowSize
//-----------------------------------------------------------------------


Size2i OSMesaWindow::getWindowSize () const
{
  JEM_ASSERT ( isOpen() );

  GLint  w, h;


  if ( Thread::getCurrent() != NativeDisplay::getThread() )
  {
    contextError ( JEM_FUNC );
  }

  OSMesaGetIntegerv ( OSMESA_WIDTH,  &w );
  OSMesaGetIntegerv ( OSMESA_HEIGHT, &h );

  return Size2i ( (GLsizei) w, (GLsizei) h );
}


//-----------------------------------------------------------------------
//   isOpen
//-----------------------------------------------------------------------


inline bool OSMesaWindow::isOpen () const
{
  return (context_ != 0);
}


//-----------------------------------------------------------------------
//   close_
//-----------------------------------------------------------------------


void OSMesaWindow::close_ ()
{
  if ( context_ )
  {
    OSMesaDestroyContext ( context_ );

    context_ = 0;
  }

  if ( buffer_ )
  {
    MemCache::dealloc ( buffer_, bufsize_ );

    buffer_  = nullptr;
    bufsize_ = 0_lint;
  }
}


//-----------------------------------------------------------------------
//   setCurrent_
//-----------------------------------------------------------------------


GLboolean OSMesaWindow::setCurrent_ ()
{
  const int     COLOR_SIZES[3] = { 1, 2, 4 };

  const GLenum  COLOR_TYPES[3] = { GL_UNSIGNED_BYTE,
                                   GL_UNSIGNED_SHORT,
                                   GL_FLOAT };

  GLboolean     result;


 again:

  // Try different color types.

  for ( int i = 0; i < 3; i++ )
  {
    ctype_   = COLOR_TYPES[i];
    bufsize_ = 4_lint * COLOR_SIZES[i] * width_ * height_;
    buffer_  = (byte*) MemCache::alloc ( bufsize_ );
    result   = OSMesaMakeCurrent ( context_, buffer_,
                                   ctype_,   width_, height_ );

    if ( result == GL_TRUE )
    {
      return GL_TRUE;
    }

    MemCache::dealloc ( buffer_, bufsize_ );

    buffer_  = nullptr;
    bufsize_ = 0_lint;
  }

  if ( width_ < 2 || height_ < 2 )
  {
    return GL_FALSE;
  }

  // Try again with a smaller image size.

  width_  /= 2;
  height_ /= 2;

  goto again;
}


//=======================================================================
//   class OSMesaDisplay
//=======================================================================


class OSMesaDisplay : public NativeDisplay
{
 public:

  typedef OSMesaDisplay     Self;
  typedef NativeDisplay     Super;


  inline                    OSMesaDisplay   ();


 protected:

  virtual                  ~OSMesaDisplay   ();

  virtual void              init_

    ( int&                    argc,
      char**                  argv )           override;

  virtual void              mainLoop_       () override;

  virtual int               getScreenDPI_   () override;

  virtual Ref<Window>       newWindow_

    ( const String&           title,
      int                     width,
      int                     height )         override;


 private:

  void                      parseGeometry_

    ( const String&           arg );


 private:

  int                       width_;
  int                       height_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


OSMesaDisplay::OSMesaDisplay ()
{
  width_ = height_ = 200;
}


OSMesaDisplay::~OSMesaDisplay ()
{}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void OSMesaDisplay::init_

  ( int&    argc,
    char**  argv )

{
  String  arg;

  int     i, k;


  i = k = 1;

  while ( i < argc )
  {
    arg = argv[i];

    if ( arg == "-geometry" )
    {
      i++;

      if ( i < argc )
      {
        parseGeometry_ ( argv[i] );

        i++;
      }

      continue;
    }

    argv[k] = argv[i];

    i++;
    k++;
  }

  for ( i = k; i < argc; i++ )
  {
    argv[i] = 0;
  }

  argc = k;
}


//-----------------------------------------------------------------------
//   mainLoop_
//-----------------------------------------------------------------------


void OSMesaDisplay::mainLoop_ ()
{
  while ( true )
  {
    popCommand_ ( 120.0 );
  }
}


//-----------------------------------------------------------------------
//   getScreenDPI_
//-----------------------------------------------------------------------


int OSMesaDisplay::getScreenDPI_ ()
{
  return 0;
}


//-----------------------------------------------------------------------
//   newWindow_
//-----------------------------------------------------------------------


Ref<NativeWindow> OSMesaDisplay::newWindow_

  ( const String&  title,
    int            width,
    int            height )

{
  if ( width < 0 )
  {
    width = width_;
  }

  if ( height < 0 )
  {
    height = height_;
  }

  return newInstance<OSMesaWindow> ( width, height );
}


//-----------------------------------------------------------------------
//   parseGeometry_
//-----------------------------------------------------------------------


void OSMesaDisplay::parseGeometry_ ( const String& arg )
{
  using jem::io::StringReader;

  Ref<StringReader>  r = newInstance<StringReader> ( arg );

  char  c;
  int   w, h;

  c = '\0';
  w = h = 0;

  try
  {
    *r >> w >> c >> h;
  }
  catch ( const Exception& )
  {
    return;
  }

  if ( c == 'x' || c == 'X' )
  {
    width_  = w;
    height_ = h;
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newOffScreenDisplay
//-----------------------------------------------------------------------


NativeDisplay* newOffScreenDisplay ()
{
  return new OSMesaDisplay ();
}


JEM_END_PACKAGE( gl )
