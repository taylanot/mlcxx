
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

#ifndef JEM_GL_FONT_H
#define JEM_GL_FONT_H

#include <jem/base/String.h>
#include <jem/gl/import.h>
#include <jem/gl/Size.h>


JEM_BEGIN_PACKAGE( gl )


class GLContext;


//-----------------------------------------------------------------------
//   enum FontSize
//-----------------------------------------------------------------------


enum FontSize
{
  SMALL_FONT,
  MEDIUM_FONT,
  LARGE_FONT
};


//-----------------------------------------------------------------------
//   enum FontType
//-----------------------------------------------------------------------


enum FontType
{
  MONOSPACE_FONT,
  PROPORTIONAL_FONT,
  SYMBOL_FONT
};


//-----------------------------------------------------------------------
//   FontID
//-----------------------------------------------------------------------


enum FontID
{
  MONOSPACE_SMALL_FONT,
  MONOSPACE_MEDIUM_FONT,
  MONOSPACE_LARGE_FONT,
  PROPORTIONAL_SMALL_FONT,
  PROPORTIONAL_MEDIUM_FONT,
  PROPORTIONAL_LARGE_FONT,
  SYMBOL_SMALL_FONT,
  SYMBOL_MEDIUM_FONT,
  SYMBOL_LARGE_FONT
};


//-----------------------------------------------------------------------
//   class Font
//-----------------------------------------------------------------------


class Font
{
 public:

  static const int          SIZE_COUNT = 3;
  static const int          TYPE_COUNT = 3;


                            Font          ();

  inline                    Font

    ( FontID                  fid )           noexcept;

  inline                    Font

    ( FontType                ftype,
      FontSize                fsize )              noexcept;

  inline                    Font

    ( const Font&             rhs )                noexcept;

  inline Font&              operator =

    ( const Font&             rhs )                noexcept;

  inline Font&              operator =

    ( FontID                  rhs )                noexcept;

  inline bool               operator ==

    ( const Font&             rhs )                noexcept;

  inline bool               operator !=

    ( const Font&             rhs )                noexcept;

  Font&                     operator ++   ()       noexcept;
  Font                      operator ++   (int)    noexcept;
  Font&                     operator --   ()       noexcept;
  Font                      operator --   (int)    noexcept;

  inline void               swap

    ( Font&                   rhs )                noexcept;

  void                      begin

    ( GLContext&              gtx );

  void                      end

    ( GLContext&              gtx );

  GLsizei                   write

    (  GLContext&              gtx,
       char                    ch );

  inline GLsizei            write

    (  GLContext&              gtx,
       const char*             str );

  inline GLsizei            write

    (  GLContext&              gtx,
       const String&           str );

  GLsizei                   write

    ( GLContext&              gtx,
      const char*             str,
      idx_t                   len );

  inline GLuint             getTexID      () const noexcept;
  inline FontID             getID         () const noexcept;
  inline FontType           getType       () const noexcept;
  inline FontSize           getSize       () const noexcept;
  GLsizei                   getHeight     () const noexcept;

  GLsizei                   getWidthOf

    ( char                    c )            const;

  inline GLsizei            getWidthOf

    ( const char*             str )          const;

  inline GLsizei            getWidthOf

    ( const String&           str )          const;

  GLsizei                   getWidthOf

    ( const char*             str,
      idx_t                   len )          const;

  Size2i                    getSizeOf

    ( char                    c )            const;

  inline Size2i             getSizeOf

    ( const char*             str )          const;

  inline Size2i             getSizeOf

    ( const String&           str )          const;

  Size2i                    getSizeOf

    ( const char*             str,
      idx_t                   len )          const;

  String                    toString      () const;

  bool                      configure

    ( const String&           name,
      const Properties&       props );

  void                      getConfig

    ( const String&           name,
      const Properties&       conf )         const;

  static Font               bestMatch

    ( FontType                ftype,
      GLsizei                 height )             noexcept;

  static Font               getDefault    ();

  static Font               getDefault

    ( FontType                ftype );

  static void               setDefault

    ( const Font&             font );


 private:

  static GLuint             initTexture_

    ( GLContext&              gtx,
      FontID                  fnID );

  GLsizei                   writeV1_

    ( GLContext&              gtx,
      char                    ch );

  GLsizei                   writeV1_

    ( GLContext&              gtx,
      const char*             str,
      idx_t                   len );

  GLsizei                   writeV3_

    ( GLContext&              gtx,
      char                    ch );

  GLsizei                   writeV3_

    ( GLContext&              gtx,
      const char*             str,
      idx_t                   len );


 private:

  class                     Data_;

  FontID                    fnID_;
  GLuint                    texID_;

};


//-----------------------------------------------------------------------
//   class FontScope
//-----------------------------------------------------------------------


class FontScope
{
 public:

  explicit inline           FontScope

    ( GLContext&              gtx,
      Font&                   font );

  inline                   ~FontScope     ();


 private:

  GLContext&                gtx_;
  Font&                     font_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 swap

  ( Font&                     lhs,
    Font&                     rhs ) noexcept;

void                        print

  ( TextOutput&               out,
    const Font&               font );



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Font
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Font::Font ( FontID fid ) noexcept
{
  fnID_  = fid;
  texID_ = 0;
}


inline Font::Font

  ( FontType  ftype,
    FontSize  fsize ) noexcept

{
  fnID_  = (FontID) (ftype * SIZE_COUNT + fsize);
  texID_ = 0;
}


inline Font::Font ( const Font& rhs ) noexcept
{
  fnID_  = rhs.fnID_;
  texID_ = rhs.texID_;
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


inline Font& Font::operator = ( const Font& rhs ) noexcept
{
  fnID_  = rhs.fnID_;
  texID_ = rhs.texID_;

  return *this;
}


inline Font& Font::operator = ( FontID rhs ) noexcept
{
  fnID_  = rhs;
  texID_ = 0;

  return *this;
}


//-----------------------------------------------------------------------
//   comparison operators
//-----------------------------------------------------------------------


inline bool Font::operator == ( const Font& rhs ) noexcept
{
  return (fnID_ == rhs.fnID_);
}


inline bool Font::operator != ( const Font& rhs ) noexcept
{
  return (fnID_ != rhs.fnID_);
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void Font::swap ( Font& rhs ) noexcept
{
  FontID  tmpID    = fnID_;
  GLuint  tmpTexID = texID_;

  fnID_      = rhs.fnID_;
  texID_     = rhs.texID_;
  rhs.fnID_  = tmpID;
  rhs.texID_ = tmpTexID;
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


inline GLsizei Font::write

  (  GLContext&   gtx,
     const char*  str )

{
  return write ( gtx, str, String::sizeOf( str ) );
}


inline GLsizei Font::write

  (  GLContext&     gtx,
     const String&  str )

{
  return write ( gtx, str.addr(), str.size() );
}


//-----------------------------------------------------------------------
//   getTexID
//-----------------------------------------------------------------------


inline GLuint Font::getTexID () const noexcept
{
  return texID_;
}


//-----------------------------------------------------------------------
//   getID
//-----------------------------------------------------------------------


inline FontID Font::getID () const noexcept
{
  return fnID_;
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline FontType Font::getType () const noexcept
{
  return (FontType) (fnID_ / SIZE_COUNT);
}


//-----------------------------------------------------------------------
//   getSize
//-----------------------------------------------------------------------


inline FontSize Font::getSize () const noexcept
{
  return (FontSize) (fnID_ % SIZE_COUNT);
}


//-----------------------------------------------------------------------
//   getWidthOf
//-----------------------------------------------------------------------


inline GLsizei Font::getWidthOf ( const char* str ) const
{
  return getWidthOf ( str, String::sizeOf( str ) );
}


inline GLsizei Font::getWidthOf ( const String& str ) const
{
  return getWidthOf ( str.addr(), str.size() );
}


//-----------------------------------------------------------------------
//   getSizeOf
//-----------------------------------------------------------------------


inline Size2i Font::getSizeOf ( const char* str ) const
{
  return getSizeOf ( str, String::sizeOf( str ) );
}


inline Size2i Font::getSizeOf ( const String& str ) const
{
  return getSizeOf ( str.addr(), str.size() );
}


//=======================================================================
//   class FontScope
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline FontScope::FontScope

  ( GLContext&   gtx,
    Font&        font ) :

    gtx_  ( gtx  ),
    font_ ( font )

{
  font_.begin ( gtx_ );
}


inline FontScope::~FontScope ()
{
  font_.end ( gtx_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void   swap

  ( Font&       lhs,
    Font&       rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( gl )

#endif
