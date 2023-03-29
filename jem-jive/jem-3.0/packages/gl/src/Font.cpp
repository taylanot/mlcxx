
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


#include <jem/base/Once.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/MemCache.h>
#include <jem/base/ReadWriteMutex.h>
#include <jem/util/Properties.h>
#include <jem/gl/Color.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLException.h>
#include <jem/gl/GLFunctions1_1.h>
#include <jem/gl/FontData.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/Font.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Font::Data_
//=======================================================================


class Font::Data_
{
 public:

  inline                    Data_        ();

  static inline Data_&      getInstance  ();


 public:

  FontID                    fnID;
  ReadWriteMutex            mutex;


 private:

  static void               init_        ();


 private:

  static Data_*             instance_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Font::Data_* Font::Data_::instance_ = nullptr;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline Font::Data_::Data_ ()
{
  fnID = PROPORTIONAL_MEDIUM_FONT;
}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


inline Font::Data_& Font::Data_::getInstance ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, &init_ );

  return *instance_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void Font::Data_::init_ ()
{
  instance_ = new Data_;
}


//=======================================================================
//   class Font
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


Font::Font ()
{
  Data_&    data = Data_::getInstance ();

  ReadLock  lock ( data.mutex );

  fnID_  = data.fnID;
  texID_ = 0;
}


//-----------------------------------------------------------------------
//   increment operators
//-----------------------------------------------------------------------


Font& Font::operator ++ () noexcept
{
  if ( getSize() != LARGE_FONT )
  {
    fnID_ = (FontID) ((int) fnID_ + 1);
  }

  return *this;
}


Font Font::operator ++ (int) noexcept
{
  Font  tmp ( fnID_ );

  if ( getSize() != LARGE_FONT )
  {
    fnID_ = (FontID) ((int) fnID_ + 1);
  }

  return tmp;
}


//-----------------------------------------------------------------------
//   decrement operators
//-----------------------------------------------------------------------


Font& Font::operator -- () noexcept
{
  if ( getSize() != SMALL_FONT )
  {
    fnID_ = (FontID) ((int) fnID_ - 1);
  }

  return *this;
}


Font Font::operator -- (int) noexcept
{
  Font  tmp ( fnID_ );

  if ( getSize() != SMALL_FONT )
  {
    fnID_ = (FontID) ((int) fnID_ - 1);
  }

  return tmp;
}


//-----------------------------------------------------------------------
//   begin
//-----------------------------------------------------------------------


void Font::begin ( GLContext& gtx )
{
  GLFuncs1_1Core*  funcs = gtx.findFunctions<GLFuncs1_1Core> ();

  texID_ = 0;

  if ( ! gtx.findGLObject( texID_, FontData::FONT_TABLE[fnID_] ) )
  {
    try
    {
      texID_ = initTexture_ ( gtx, fnID_ );
    }
    catch ( const GLException& ex )
    {
      texID_ = 0;

      print ( System::warn(), "Failed to create a font texture: ",
              ex.what(), '\n' );
    }
  }

  gtx.enableAlphaTest ( true );

  if ( texID_ && funcs &&
       ! gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    funcs->glEnable      ( GL_TEXTURE_2D );
    funcs->glBindTexture ( GL_TEXTURE_2D, texID_ );
  }
}


//-----------------------------------------------------------------------
//   end
//-----------------------------------------------------------------------


void Font::end ( GLContext& gtx )
{
  GLFuncs1_1Core*  funcs = gtx.findFunctions<GLFuncs1_1Core> ();

  gtx.enableAlphaTest ( false );

  if ( funcs && ! gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    funcs->glDisable     ( GL_TEXTURE_2D );
    funcs->glBindTexture ( GL_TEXTURE_2D, 0 );
  }

  texID_ = 0;

  checkGLError ( gtx, JEM_FUNC );
}


//-----------------------------------------------------------------------
//   write (one character)
//-----------------------------------------------------------------------


GLsizei Font::write

  ( GLContext&  gtx,
    char        ch )

{
  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    return writeV3_ ( gtx, ch );
  }
  else
  {
    return writeV1_ ( gtx, ch );
  }
}


//-----------------------------------------------------------------------
//   write (a string)
//-----------------------------------------------------------------------


GLsizei Font::write

  ( GLContext&   gtx,
    const char*  str,
    idx_t        len )

{
  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    return writeV3_ ( gtx, str, len );
  }
  else
  {
    return writeV1_ ( gtx, str, len );
  }
}


//-----------------------------------------------------------------------
//   getHeight
//-----------------------------------------------------------------------


GLsizei Font::getHeight () const noexcept
{
  const GLubyte*  data = FontData::FONT_TABLE[fnID_];

  return (GLsizei) data[2];
}


//-----------------------------------------------------------------------
//   getWidthOf
//-----------------------------------------------------------------------


GLsizei Font::getWidthOf ( char ch ) const
{
  const GLubyte*  chData =

    FontData::getCharData ( ch, FontData::FONT_TABLE[fnID_] );

  return (GLsizei) chData[2];
}


GLsizei Font::getWidthOf

  ( const char*  str,
    idx_t        len ) const

{
  const GLubyte*  fnData = FontData::FONT_TABLE[fnID_];
  const GLubyte*  chData = nullptr;

  GLsizei         width;

  if ( getType() == MONOSPACE_FONT )
  {
    chData = FontData::getCharData ( '0', fnData );
    width  = (GLsizei) len * (GLsizei) chData[2];
  }
  else
  {
    width = 0;

    for ( idx_t i = 0; i < len; i++ )
    {
      chData = FontData::getCharData ( str[i], fnData );
      width += (GLsizei) chData[2];
    }
  }

  return width;
}


//-----------------------------------------------------------------------
//   getSizeOf
//-----------------------------------------------------------------------


Size2i Font::getSizeOf ( char ch ) const
{
  const GLubyte*  fnData = FontData::FONT_TABLE  [ fnID_ ];
  const GLubyte*  chData = FontData::getCharData ( ch, fnData );

  return Size2i ( (GLsizei) chData[2], (GLsizei) fnData[2] );
}


Size2i Font::getSizeOf

  ( const char*  str,
    idx_t        len ) const

{
  const GLubyte*  fnData = FontData::FONT_TABLE[fnID_];

  return Size2i ( getWidthOf( str, len ), (GLsizei) fnData[2] );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Font::toString () const
{
  FontType  fnType = getType ();
  FontSize  fnSize = getSize ();

  return String::format ( "%s %s",
                          FontData::TYPE_NAMES[fnType],
                          FontData::SIZE_NAMES[fnSize] );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


bool Font::configure

  ( const String&      name,
    const Properties&  props )

{
  if ( props.contains( name ) )
  {
    FontType  fnType = getType ();
    FontSize  fnSize = getSize ();

    String    fontName;
    String    typeName;
    String    sizeName;
    idx_t     k;

    props.get ( fontName, name );

    fontName = fontName.stripWhite ();
    k        = fontName.find  ( ' ' );

    if ( k >= 0 )
    {
      typeName = fontName[slice(BEGIN,k)]  .stripWhite ();
      sizeName = fontName[slice(k + 1,END)].stripWhite ();
    }
    else
    {
      typeName = fontName;
    }

    typeName = typeName.toLower ();
    sizeName = sizeName.toLower ();

    for ( int i = 0; i < SIZE_COUNT; i++ )
    {
      if ( typeName == FontData::SIZE_NAMES[i] )
      {
        typeName.swap ( sizeName );
        break;
      }
    }

    if ( typeName.size() > 0 )
    {
      for ( int i = 0; i < TYPE_COUNT; i++ )
      {
        if ( typeName == FontData::TYPE_NAMES[i] )
        {
          fnType = (FontType) i;
          break;
        }

        if ( i >= TYPE_COUNT )
        {
          props.propertyError (
            name,
            String::format (
              "unknown font type: %s; must be "
              "`%s\', `%s\' or `%s\'",
              typeName,
              FontData::TYPE_NAMES[MONOSPACE_FONT],
              FontData::TYPE_NAMES[PROPORTIONAL_FONT],
              FontData::TYPE_NAMES[SYMBOL_FONT]
            )
          );
        }
      }
    }

    if ( sizeName.size() > 0 )
    {
      for ( int i = 0; i < SIZE_COUNT; i++ )
      {
        if ( typeName == FontData::SIZE_NAMES[i] )
        {
          fnSize = (FontSize) i;
          break;
        }

        if ( i >= SIZE_COUNT )
        {
          props.propertyError (
            name,
            String::format (
              "unknown font size: %s; must be "
              "`%s\', `%s\' or `%s\'",
              sizeName,
              FontData::SIZE_NAMES[SMALL_FONT],
              FontData::SIZE_NAMES[MEDIUM_FONT],
              FontData::SIZE_NAMES[LARGE_FONT]
            )
          );
        }
      }
    }

    fnID_ = (FontID) (fnType * SIZE_COUNT + fnSize);

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Font::getConfig

  ( const String&      name,
    const Properties&  conf ) const

{
  conf.set ( name, toString() );
}


//-----------------------------------------------------------------------
//   bestMatch
//-----------------------------------------------------------------------


Font Font::bestMatch

  ( FontType  ftype,
    GLsizei   height ) noexcept

{
  int      large = SIZE_COUNT * ftype + LARGE_FONT;
  int      small = SIZE_COUNT * ftype + SMALL_FONT;
  int      best  = small;
  GLsizei  diff  = height;

  for ( int i = large; i >= small; i-- )
  {
    const GLubyte*  fnData = FontData::FONT_TABLE[i];
    GLsizei         delta  = height - (GLsizei) fnData[2];

    if ( delta < 0 )
    {
      delta = -delta;
    }

    if ( delta <= diff )
    {
      diff = delta;
      best = i;
    }
  }

  return Font ( (FontID) best );
}


//-----------------------------------------------------------------------
//   getDefault
//-----------------------------------------------------------------------


Font Font::getDefault ()
{
  FontID  fid;

  {
    Data_&    data = Data_::getInstance ();

    ReadLock  lock ( data.mutex );

    fid = data.fnID;
  }

  return Font ( fid );
}


Font Font::getDefault ( FontType ftype )
{
  Font  fn = getDefault ();

  return Font ( ftype, fn.getSize() );
}


//-----------------------------------------------------------------------
//   setDefault
//-----------------------------------------------------------------------


void Font::setDefault ( const Font& font )
{
  Data_&     data = Data_::getInstance ();

  WriteLock  lock ( data.mutex );

  data.fnID = font.fnID_;
}


//-----------------------------------------------------------------------
//   initTexture_
//-----------------------------------------------------------------------


GLuint Font::initTexture_

  ( GLContext&  gtx,
    FontID      fnID )

{
  const GLubyte*   fnData = FontData::FONT_TABLE [ fnID   ];
  const GLubyte*   txData = FontData::getTexData ( fnData );

  GLFuncs1_1Core*  funcs  = gtx.findFunctions<GLFuncs1_1Core> ();

  GLubyte*         image;
  GLsizei          width;
  GLsizei          height;
  GLuint           texID;
  GLenum           err;
  GLint            i, j, n, w;


  if ( ! funcs )
  {
    return 0;
  }

  if ( txData[0] != FontData::MAGIC_BYTE )
  {
    throw Error ( JEM_FUNC, "invalid texture data" );
  }

  txData++;

  texID = gtx.newTexture ( fnData, nullptr );

  funcs->glBindTexture ( GL_TEXTURE_2D, texID );

  // Check whether the texture can be accommodated.

  width  = (GLsizei) (1 << fnData[0]);
  height = (GLsizei) (1 << fnData[1]);

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    // The alpha channel does not exist for version 3.3 and later.

    funcs->glTexImage ( GL_PROXY_TEXTURE_2D, 0, GL_R8,
                        width, height,       0, GL_RED,
                        GL_UNSIGNED_BYTE,    nullptr );
  }
  else
  {
    funcs->glTexImage ( GL_PROXY_TEXTURE_2D, 0, GL_ALPHA8,
                        width, height,       0, GL_ALPHA,
                        GL_UNSIGNED_BYTE,    nullptr );
  }

  funcs->glGetTexLevelParameter ( GL_PROXY_TEXTURE_2D, 0,
                                  GL_TEXTURE_WIDTH, &w );

  if ( w == 0 )
  {
    gtx.freeGLObject ( fnData );

    throw GLException ( JEM_FUNC, "not enough texture memory" );
  }

  // Create the texture image. The texture data is stored in a
  // compressed format based on run-length encoding.

  image = (GLubyte*) MemCache::alloc ( (size_t) (width * height) );

  for ( i = 0; i < height; i++ )
  {
    j = 0;

    while ( j < width )
    {
      n = j + ((GLint) txData[0]) + 1;

      for ( ; j < n; j++ )
      {
        image[i * width + j] = txData[1];
      }

      txData += 2;
    }

    JEM_ASSERT ( j == width );
  }

  if ( txData[0] != FontData::MAGIC_BYTE )
  {
    gtx.freeGLObject  ( fnData );
    MemCache::dealloc ( image, (size_t) (width * height) );

    throw Error ( JEM_FUNC, "invalid texture data" );
  }

  // Set the alpha func in the case that has not been done yet.

  gtx.setAlphaTest ( 0.05F );

  // Define the texture.

  funcs->glTexParameter ( GL_TEXTURE_2D,
                          GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  funcs->glTexParameter ( GL_TEXTURE_2D,
                          GL_TEXTURE_MIN_FILTER, GL_LINEAR );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    // The alpha channel does not exist in version 3.3 and later;
    // swizzle the alpha channel to red.

    GLint swizzleMask[] = { GL_ZERO, GL_ZERO, GL_ZERO, GL_RED };

    funcs->glTexParameter ( GL_TEXTURE_2D,
                            GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    funcs->glTexParameter ( GL_TEXTURE_2D,
                            GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    funcs->glTexParameter ( GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA,
                            swizzleMask );
    funcs->glTexImage     ( GL_TEXTURE_2D, 0, GL_R8,
                            width, height, 0, GL_RED,
                            GL_UNSIGNED_BYTE, image );
  }
  else
  {
    GLFuncs1_1*  funcs1 = gtx.findFunctions<GLFuncs1_1> ();

    funcs->glTexParameter ( GL_TEXTURE_2D,
                            GL_TEXTURE_WRAP_S, GL_CLAMP );
    funcs->glTexParameter ( GL_TEXTURE_2D,
                            GL_TEXTURE_WRAP_T, GL_CLAMP );

    if ( funcs1 )
    {
      funcs1->glTexEnv    ( GL_TEXTURE_ENV,
                            GL_TEXTURE_ENV_MODE, GL_MODULATE );
    }

    funcs->glTexImage     ( GL_TEXTURE_2D, 0, GL_ALPHA8,
                            width, height, 0, GL_ALPHA,
                            GL_UNSIGNED_BYTE, image );
  }

  MemCache::dealloc ( image, (size_t) (width * height) );

  if ( (err = funcs->glGetError()) != GL_NO_ERROR )
  {
    gtx.freeGLObject ( fnData );

    throw GLException ( JEM_FUNC, gtx.errorString( err ) );
  }

  return texID;
}


//-----------------------------------------------------------------------
//   writeV1_ (one character)
//-----------------------------------------------------------------------


GLsizei Font::writeV1_

  ( GLContext&  gtx,
    char        ch )

{
  const GLubyte*  fnData = FontData::FONT_TABLE  [ fnID_ ];
  const GLubyte*  chData = FontData::getCharData ( ch, fnData );

  const GLfloat   xScale = 1.0F / (GLfloat) (1 << fnData[0]);
  const GLfloat   yScale = 1.0F / (GLfloat) (1 << fnData[1]);

  GLFuncs1*       funcs  = gtx.findFunctions<GLFuncs1> ();

  GLfloat  tx1, tx2;
  GLfloat  ty1, ty2;
  GLint    vx1, vx2;
  GLint    vy1, vy2;


  if ( ! funcs )
  {
    return 0;
  }

  tx1 = xScale * (GLfloat) (chData[0]);
  tx2 = xScale * (GLfloat) (chData[0] + chData[2]);
  ty1 = yScale * (GLfloat) (chData[1]);
  ty2 = yScale * (GLfloat) (chData[1] + fnData[2]);

  vx1 = 0;
  vx2 = (GLint) chData[2];
  vy1 = 0;
  vy2 = (GLint) fnData[2];

  funcs->glBegin ( GL_QUADS );
  {
    funcs->glTexCoord ( tx1, ty1 );
    funcs->glVertex   ( vx1, vy1 );
    funcs->glTexCoord ( tx2, ty1 );
    funcs->glVertex   ( vx2, vy1 );
    funcs->glTexCoord ( tx2, ty2 );
    funcs->glVertex   ( vx2, vy2 );
    funcs->glTexCoord ( tx1, ty2 );
    funcs->glVertex   ( vx1, vy2 );
  }
  funcs->glEnd ();

  return (GLsizei) vx2;
}


//-----------------------------------------------------------------------
//   writeV1_ (a string)
//-----------------------------------------------------------------------


GLsizei Font::writeV1_

  ( GLContext&   gtx,
    const char*  str,
    idx_t        len )

{
  const GLubyte*  fnData = FontData::FONT_TABLE[fnID_];
  const GLubyte*  chData = nullptr;

  const GLfloat   xScale = 1.0F / (GLfloat) (1 << fnData[0]);
  const GLfloat   yScale = 1.0F / (GLfloat) (1 << fnData[1]);

  GLFuncs1*       funcs  = gtx.findFunctions<GLFuncs1> ();

  GLfloat  tx1, tx2;
  GLfloat  ty1, ty2;
  GLint    vx1, vx2;
  GLint    vy1, vy2;


  if ( ! funcs )
  {
    return 0;
  }

  vx1 = 0;
  vx2 = 0;
  vy1 = 0;
  vy2 = (GLint) fnData[2];

  funcs->glBegin ( GL_QUADS );

  for ( idx_t i = 0; i < len; i++ )
  {
    chData = FontData::getCharData ( str[i], fnData );

    tx1 = xScale * (GLfloat) (chData[0]);
    tx2 = xScale * (GLfloat) (chData[0] + chData[2]);
    ty1 = yScale * (GLfloat) (chData[1]);
    ty2 = yScale * (GLfloat) (chData[1] + fnData[2]);

    vx1 = vx2;
    vx2 = vx2 + (GLint) chData[2];

    funcs->glTexCoord ( tx1, ty1 );
    funcs->glVertex   ( vx1, vy1 );
    funcs->glTexCoord ( tx2, ty1 );
    funcs->glVertex   ( vx2, vy1 );
    funcs->glTexCoord ( tx2, ty2 );
    funcs->glVertex   ( vx2, vy2 );
    funcs->glTexCoord ( tx1, ty2 );
    funcs->glVertex   ( vx1, vy2 );
  }

  funcs->glEnd ();

  return (GLsizei) vx2;
}


//-----------------------------------------------------------------------
//   writeV3_ (one character)
//-----------------------------------------------------------------------


GLsizei Font::writeV3_

  ( GLContext&  gtx,
    char        ch )

{
  const GLubyte*  fnData     = FontData::FONT_TABLE  [ fnID_ ];
  const GLubyte*  chData     = FontData::getCharData ( ch, fnData );

  const GLfloat   xScale     = 1.0F / (GLfloat) (1 << fnData[0]);
  const GLfloat   yScale     = 1.0F / (GLfloat) (1 << fnData[1]);

  GLfloat         vertXY[12] = { 0.0F };
  GLfloat         vertUV [8];

  GLfloat        tx1, tx2;
  GLfloat        ty1, ty2;
  GLfloat        vx1, vx2;
  GLfloat        vy1, vy2;

  tx1       = xScale * (GLfloat) (chData[0]);
  tx2       = xScale * (GLfloat) (chData[0] + chData[2]);
  ty1       = yScale * (GLfloat) (chData[1]);
  ty2       = yScale * (GLfloat) (chData[1] + fnData[2]);

  vx1       = 0.0F;
  vx2       = (GLfloat) chData[2];
  vy1       = 0.0F;
  vy2       = (GLfloat) fnData[2];

  vertXY[0] = vx1; vertXY[ 1] = vy1;
  vertXY[3] = vx2; vertXY[ 4] = vy1;
  vertXY[6] = vx2; vertXY[ 7] = vy2;
  vertXY[9] = vx1; vertXY[10] = vy2;

  vertUV[0] = tx1; vertUV[1]  = ty1;
  vertUV[2] = tx2; vertUV[3]  = ty1;
  vertUV[4] = tx2; vertUV[5]  = ty2;
  vertUV[6] = tx1; vertUV[7]  = ty2;

  ShaderDraw::drawTex2D ( gtx, 12, vertXY, vertUV, texID_,
                          gtx.getColor (),
                          ShaderDraw::TRIANGLE_FAN );

  return (GLsizei) vx2;
}


//-----------------------------------------------------------------------
//   writeV3_ (a string)
//-----------------------------------------------------------------------


GLsizei Font::writeV3_

  ( GLContext&   gtx,
    const char*  str,
    idx_t        len )

{
  const GLubyte*  fnData = FontData::FONT_TABLE[fnID_];
  const GLubyte*  chData = nullptr;

  const GLfloat   xScale = 1.0F / (GLfloat) (1 << fnData[0]);
  const GLfloat   yScale = 1.0F / (GLfloat) (1 << fnData[1]);

  GLfloatVector   buffer ( (18 + 12) * len );
  GLfloatVector   vertXY = buffer[slice(BEGIN,18 * len)];
  GLfloatVector   vertUV = buffer[slice(18 * len,END)];

  GLfloat         tx1, tx2;
  GLfloat         ty1, ty2;
  GLfloat         vx1, vx2;
  GLfloat         vy1, vy2;


  vx1    = 0.0F;
  vx2    = 0.0F;
  vy1    = 0.0F;
  vy2    = (GLfloat) fnData[2];

  vertXY = 0.0F;

  for ( idx_t i = 0; i < len; i++ )
  {
    idx_t  j;

    chData = FontData::getCharData ( str[i], fnData );

    tx1 = xScale * (GLfloat) (chData[0]);
    tx2 = xScale * (GLfloat) (chData[0] + chData[2]);
    ty1 = yScale * (GLfloat) (chData[1]);
    ty2 = yScale * (GLfloat) (chData[1] + fnData[2]);

    vx1 = vx2;
    vx2 = vx2 + (GLfloat) chData[2];

    j   = 18 * i;

    vertXY[j]      = vx1; vertXY[j +  1] = vy1;
    vertXY[j +  3] = vx2; vertXY[j +  4] = vy1;
    vertXY[j +  6] = vx2; vertXY[j +  7] = vy2;

    vertXY[j +  9] = vx2; vertXY[j + 10] = vy2;
    vertXY[j + 12] = vx1; vertXY[j + 13] = vy2;
    vertXY[j + 15] = vx1; vertXY[j + 16] = vy1;

    j = 12 * i;

    vertUV[j]      = tx1; vertUV[j +  1] = ty1;
    vertUV[j +  2] = tx2; vertUV[j +  3] = ty1;
    vertUV[j +  4] = tx2; vertUV[j +  5] = ty2;

    vertUV[j +  6] = tx2; vertUV[j +  7] = ty2;
    vertUV[j +  8] = tx1; vertUV[j +  9] = ty2;
    vertUV[j + 10] = tx1; vertUV[j + 11] = ty1;
  }

  ShaderDraw::drawTex2D ( gtx, vertXY, vertUV, texID_,
                          gtx.getColor (),
                          ShaderDraw::TRIANGLES );

  return (GLsizei) vx2;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void            print

  ( TextOutput&   out,
    const Font&   font )

{
  FontType  fnType = font.getType ();
  FontSize  fnSize = font.getSize ();

  print ( out, FontData::TYPE_NAMES[fnType], ' ' );
  print ( out, FontData::SIZE_NAMES[fnSize] );
}


JEM_END_PACKAGE( gl )
