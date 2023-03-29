
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

#ifndef JEM_GL_COLOR_H
#define JEM_GL_COLOR_H

#include <jem/base/assert.h>
#include <jem/base/String.h>
#include <jem/gl/gl.h>
#include <jem/gl/import.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Color
//-----------------------------------------------------------------------


class Color
{
 public:

  struct                    Def
  {
    const char*               name;
    Color                   (*color) ();
  };

  static const Def          DEFS[];
  static const GLfloat      OPAQUE_ALPHA;


  explicit inline           Color

    ( GLfloat                 c = 0.0 )            noexcept;

  explicit inline           Color

    ( const GLfloat*          c )                  noexcept;

  inline                    Color

    ( GLfloat                 r,
      GLfloat                 g,
      GLfloat                 b,
      GLfloat                 a = 1.0 )            noexcept;

  inline                    Color

    ( const Color&            rhs )                noexcept;

  inline Color&             operator =

    ( const Color&            rhs )                noexcept;

  template <class I>
    inline GLfloat&         operator []

    ( I                       i );

  template <class I>
    inline GLfloat          operator []

    ( I                       i )            const;

  inline GLfloat            r             () const noexcept;
  inline GLfloat            g             () const noexcept;
  inline GLfloat            b             () const noexcept;
  inline GLfloat            a             () const noexcept;

  inline bool               isOpaque      () const noexcept;
  inline       GLfloat*     addr          ()       noexcept;
  inline const GLfloat*     addr          () const noexcept;

  static Color              black         ()       noexcept;
  static Color              blue          ()       noexcept;
  static Color              brown         ()       noexcept;
  static Color              cyan          ()       noexcept;
  static Color              gold          ()       noexcept;
  static Color              gray          ()       noexcept;
  static Color              green         ()       noexcept;
  static Color              magenta       ()       noexcept;
  static Color              orange        ()       noexcept;
  static Color              pink          ()       noexcept;
  static Color              purple        ()       noexcept;
  static Color              red           ()       noexcept;
  static Color              shadow        ()       noexcept;
  static Color              silver        ()       noexcept;
  static Color              steel         ()       noexcept;
  static Color              titanium      ()       noexcept;
  static Color              violet        ()       noexcept;
  static Color              white         ()       noexcept;
  static Color              yellow        ()       noexcept;
  static Color              random        ();

  static Color              parse

    ( const String&           str );

  static Color              byName

    ( const String&           name );

  static String             getChoice     ();

  bool                      findName

    ( String&                 name,
      double                  eps = 0.01 )   const;

  String                    toString      () const;

   bool                     configure

    ( const String&           name,
      const Properties&       props );

  void                      getConfig

    ( const String&           name,
      const Properties&       conf )         const;


 private:

  GLfloat                   data_[4];

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


void                        read

  ( io::TextInput&            in,
    Color&                    c );

template <class Output>

  inline void               print

  ( Output&                   out,
    const Color&              c );

bool                        clamp

  ( Color&                    c )      noexcept;

inline Color                clamped

  ( const Color&              c )      noexcept;

inline bool                 isOpaque

  ( GLfloat                   alpha )  noexcept;



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Color
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Color::Color ( GLfloat c ) noexcept
{
  data_[0] = data_[1] = data_[2] = c;
  data_[3] = 1.0;
}


inline Color::Color ( const GLfloat* c ) noexcept
{
  data_[0] = c[0];
  data_[1] = c[1];
  data_[2] = c[2];
  data_[3] = c[3];
}


inline Color::Color ( GLfloat r, GLfloat g,
                      GLfloat b, GLfloat a ) noexcept
{
  data_[0] = r;
  data_[1] = g;
  data_[2] = b;
  data_[3] = a;
}


inline Color::Color ( const Color& rhs ) noexcept
{
  data_[0] = rhs.data_[0];
  data_[1] = rhs.data_[1];
  data_[2] = rhs.data_[2];
  data_[3] = rhs.data_[3];
}


inline Color& Color::operator = ( const Color& rhs ) noexcept
{
  data_[0] = rhs.data_[0];
  data_[1] = rhs.data_[1];
  data_[2] = rhs.data_[2];
  data_[3] = rhs.data_[3];

  return *this;
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class I>

  inline GLfloat& Color::operator [] ( I i )

{
  JEM_ASSERT ( checkIndex( i, 4 ) );

  return data_[i];
}


template <class I>

  inline GLfloat Color::operator [] ( I i ) const

{
  JEM_ASSERT ( checkIndex( i, 4 ) );

  return data_[i];
}


//-----------------------------------------------------------------------
//   r, g, b, a
//-----------------------------------------------------------------------


inline GLfloat Color::r () const noexcept
{
  return data_[0];
}


inline GLfloat Color::g () const noexcept
{
  return data_[1];
}


inline GLfloat Color::b () const noexcept
{
  return data_[2];
}


inline GLfloat Color::a () const noexcept
{
  return data_[3];
}


//-----------------------------------------------------------------------
//   isOpaque
//-----------------------------------------------------------------------


inline bool Color::isOpaque () const noexcept
{
  return (data_[3] > OPAQUE_ALPHA);
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline GLfloat* Color::addr () noexcept
{
  return data_;
}


inline const GLfloat* Color::addr () const noexcept
{
  return data_;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void     print

  ( Output&         out,
    const Color&    c )

{
  print ( out, c.toString() );
}


//-----------------------------------------------------------------------
//   clamped
//-----------------------------------------------------------------------


inline Color clamped ( const Color& c ) noexcept
{
  Color  tmp = c;

  clamp ( tmp );

  return tmp;
}


//-----------------------------------------------------------------------
//   isOpaque
//-----------------------------------------------------------------------


inline bool isOpaque ( GLfloat alpha ) noexcept
{
  return (alpha > Color::OPAQUE_ALPHA);
}


JEM_END_PACKAGE( gl )

#endif
