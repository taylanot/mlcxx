
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/Array.h>
#include <jem/io/utilities.h>
#include <jem/io/CRCOutputStream.h>
#include <jem/io/ZipOutputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/gl/Rect.h>
#include <jem/gl/GrabScope.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/PNGGrabber.h>


JEM_DEFINE_CLASS( jem::gl::PNGGrabber );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class PNGGrabber
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PNGGrabber::PNGGrabber ()
{}


PNGGrabber::~PNGGrabber ()
{}


//-----------------------------------------------------------------------
//   grabFrame
//-----------------------------------------------------------------------


void PNGGrabber::grabFrame

  ( OutputStream&   out,
    RenderContext&  rtx,
    GfxNode&        scene,
    Rect&           rect )

{
  GrabScope  scope ( rtx.getGLContext() );

  scope.drawFrame  ( rtx, scene );
  scope.adjustRect ( rect );
  saveFrame        ( out, rect );
}


//-----------------------------------------------------------------------
//   saveFrame
//-----------------------------------------------------------------------


void PNGGrabber::saveFrame

  ( OutputStream&  out,
    const Rect&    rect )

{
  using jem::io::encodeUL32;
  using jem::io::CRCOutputStream;

  JEM_PRECHECK ( rect.isValid() );

  Ref<CRCOutputStream>  crc  =

    newInstance<CRCOutputStream> ( & out );

  const byte   PNG_HEADER[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };

  Array<byte>  image;
  byte         buf[32];
  int          i = 0;


  encodeFrame ( image, rect );

  // The PNG header

  out.write   ( PNG_HEADER, sizeof(PNG_HEADER) );

  // The IHDR chunk

  encodeUL32  ( buf, 13_ulint );  // Length of IHDR data = 13
  out.write   ( buf, 4    );

  buf[i++] = 73;  // 'I'
  buf[i++] = 72;  // 'H'
  buf[i++] = 68;  // 'D'
  buf[i++] = 82;  // 'R'

  encodeUL32 ( buf + i, (ulint) rect.width  ); i += 4;
  encodeUL32 ( buf + i, (ulint) rect.height ); i += 4;

  buf[i++] =  8;  // Bit depth
  buf[i++] =  2;  // Color type (RGB)
  buf[i++] =  0;  // Compression method
  buf[i++] =  0;  // Filter method
  buf[i++] =  0;  // Interlace method

  crc->write    ( buf, i );
  crc->flushCRC ();

  // The IDAT chunk containing the image

  encodeUL32    ( buf, (ulint) image.size() );
  out.write     ( buf, 4 );

  i = 0;

  buf[i++] = 73;  // 'I'
  buf[i++] = 68;  // 'D'
  buf[i++] = 65;  // 'A'
  buf[i++] = 84;  // 'T'

  crc->write    ( buf, 4 );
  crc->write    ( image.addr(), image.size() );
  crc->flushCRC ();

  // The IEND chunk

  encodeUL32 ( buf, 0_ulint );  // Length of IEND data = 0
  out.write  ( buf, 4 );

  i = 0;

  buf[i++] = 73; // 'I'
  buf[i++] = 69; // 'E'
  buf[i++] = 78; // 'N'
  buf[i++] = 68; // 'D'

  crc->write    ( buf, 4 );
  crc->flushCRC ();
  crc->flush    ();

  crc = nullptr;
}


//-----------------------------------------------------------------------
//   encodeFrame
//-----------------------------------------------------------------------


void PNGGrabber::encodeFrame

  ( Array<byte>&  image,
    const Rect&   rect )

{
  using jem::io::ZipOutputStream;
  using jem::io::ArrayOutputStream;

  Ref<ArrayOutputStream>  zbuf   =

    newInstance<ArrayOutputStream> ();

  Ref<ZipOutputStream>    zipper =

    newInstance<ZipOutputStream> ( zbuf );


  const byte   FILTER_TYPE[1] = { 0x0 };

  const idx_t  w = (idx_t) rect.width;
  const idx_t  h = (idx_t) rect.height;

  Array<byte>  fbuf ( 3 * w * h );


  glReadPixels ( rect.x, rect.y, rect.width, rect.height,
                 GL_RGB, GL_UNSIGNED_BYTE, fbuf.addr() );

  for ( idx_t i = h - 1; i >= 0; i-- )
  {
    zipper->write ( FILTER_TYPE, 1 );
    zipper->write ( fbuf.addr( i * 3 * w ), 3 * w );
  }

  zipper->flush ();
  zipper = nullptr;

  image.ref ( zbuf->toArray() );
}


JEM_END_PACKAGE( gl )
