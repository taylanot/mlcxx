
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


#include <cmath>
#include <cctype>
#include <cstdlib>
#include <jem/base/rtti.h>
#include <jem/base/Float.h>
#include <jem/base/Integer.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/Array.h>
#include <jem/io/TextInput.h>
#include <jem/util/Dictionary.h>
#include <jem/util/Properties.h>
#include <jem/util/PropertyException.h>
#include <jem/gl/Color.h>


JEM_BEGIN_PACKAGE( gl )


using jem::io::TextInput;
using jem::io::TextOutput;
using jem::util::PropertyException;


//=======================================================================
//   class Color
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const Color::Def Color::DEFS[] =

  { { "black",    black    },
    { "blue",     blue     },
    { "brown",    brown    },
    { "cyan",     cyan     },
    { "gold",     gold     },
    { "gray",     gray     },
    { "green",    green    },
    { "magenta",  magenta  },
    { "orange",   orange   },
    { "pink",     pink     },
    { "purple",   purple   },
    { "red",      red      },
    { "silver",   silver   },
    { "steel",    steel    },
    { "titanium", titanium },
    { "violet",   violet   },
    { "white",    white    },
    { "yellow",   yellow   },
    { 0,          0        } };

const GLfloat    Color::OPAQUE_ALPHA = 0.95F;


//-----------------------------------------------------------------------
//   black, blue, brown, etc.
//-----------------------------------------------------------------------


Color Color::black () noexcept
{
  return Color ( 0.0F, 0.0F, 0.0F );
}


Color Color::blue () noexcept
{
  return Color ( 0.0F, 0.0F, 1.0F );
}


Color Color::brown () noexcept
{
  return Color ( 0.647F, 0.165F, 0.165F );
}


Color Color::cyan () noexcept
{
  return Color ( 0.0F, 1.0F, 1.0F );
}


Color Color::gold () noexcept
{
  return Color ( 1.0F, 0.843F, 0.0F );
}


Color Color::gray () noexcept
{
  return Color ( 0.745F, 0.745F, 0.745F );
}


Color Color::green () noexcept
{
  return Color ( 0.0F, 1.0F, 0.0F );
}


Color Color::magenta () noexcept
{
  return Color ( 1.0F, 0.0F, 1.0F );
}


Color Color::orange () noexcept
{
  return Color ( 1.0F, 0.647F, 0.0F );
}


Color Color::pink () noexcept
{
  return Color ( 1.0F, 0.753F, 0.796F );
}


Color Color::purple () noexcept
{
  return Color ( 0.627F, 0.125F, 0.941F );
}


Color Color::red () noexcept
{
  return Color ( 1.0F, 0.0F, 0.0F );
}


Color Color::shadow () noexcept
{
  return Color ( 0.3F, 0.3F, 0.3F, 0.5F );
}


Color Color::silver () noexcept
{
  return Color ( 0.465F, 0.531F, 0.598F );
}


Color Color::steel () noexcept
{
  return Color ( 0.275F, 0.510F, 0.706F );
}


Color Color::titanium () noexcept
{
  return Color ( 0.438F, 0.500F, 0.563F );
}


Color Color::violet () noexcept
{
  return Color ( 0.539F, 0.168F, 0.883F );
}


Color Color::white () noexcept
{
  return Color ( 1.0F, 1.0F, 1.0F );
}


Color Color::yellow () noexcept
{
  return Color ( 1.0F, 1.0F, 0.0F );
}


//-----------------------------------------------------------------------
//   random
//-----------------------------------------------------------------------


Color Color::random ()
{
  return Color (
    (GLfloat) ::rand() / (GLfloat) RAND_MAX,
    (GLfloat) ::rand() / (GLfloat) RAND_MAX,
    (GLfloat) ::rand() / (GLfloat) RAND_MAX
  );
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


Color Color::parse ( const String& str )
{
  String       s = str.stripWhite ();

  const idx_t  n = s.size ();

  Color        c;
  double       x;
  idx_t        i, j;
  int          k;


  if ( n == 0 )
  {
    throw ParseException ( JEM_FUNC, "empty string" );
  }

  if ( s[0] != '[' )
  {
    try
    {
      c = byName ( s );
    }
    catch ( const IllegalInputException& ex )
    {
      throw ParseException ( JEM_FUNC, ex.what() );
    }

    return c;
  }

  c[3] = 1.0;
  i    = 0;

  for ( k = 0; k < 4; k++ )
  {
    j = i + 1;

    for ( i = j; i < n && s[i] != ',' && s[i] != ']'; i++ ) ;

    if ( i >= n )
    {
      break;
    }

    if ( i == j )
    {
      throw ParseException ( JEM_FUNC, "missing color component" );
    }

    if ( ! Float::parse( x, s[slice(j,i)] ) )
    {
      throw ParseException (
        JEM_FUNC,
        "invalid color component; expected a float"
      );
    }

    if ( x < 0.0 || x > 1.0 )
    {
      throw ParseException (
        JEM_FUNC,
        "color component out of range; valid range is [0,1]"
      );
    }

    c[k] = (GLfloat) x;

    if ( s[i] == ']' )
    {
      break;
    }
  }

  if ( k < 2 )
  {
    throw ParseException (
      JEM_FUNC,
      "incomplete color vector; "
      "at least three components are required"
    );
  }

  if ( i >= n || s[i] != ']' )
  {
    throw ParseException (
      JEM_FUNC,
      "invalid color vector: missing `]\'"
    );
  }

  for ( i++; i < n && std::isspace(s[i]); i++ ) ;

  if ( i != n )
  {
    throw ParseException (
      JEM_FUNC,
      "invalid color vector: parse error after `]\'"
    );
  }

  return c;
}


//-----------------------------------------------------------------------
//   byName
//-----------------------------------------------------------------------


Color Color::byName ( const String& cname )
{
  String       name = cname.toLower ();

  const idx_t  len  = name.size ();

  Color        c;
  GLfloat      fscale;
  idx_t        iscale;
  int          i;


  iscale = 100;

  for ( idx_t j = 0; j < len; j++ )
  {
    if ( std::isdigit( name[j] ) )
    {
      if ( Integer::parse( iscale, name[slice(j,END)] ) )
      {
        name = name[slice(BEGIN,j)].stripWhite ();
      }

      break;
    }
  }

  for ( i = 0; DEFS[i].name; i++ )
  {
    if ( DEFS[i].name == name )
    {
      c = DEFS[i].color ();
      break;
    }
  }

  if ( ! DEFS[i].name )
  {
    String  choice = getChoice ();

    throw IllegalInputException (
      JEM_FUNC,
      String::format (
        "unknown color: `%s\'; should be %s",
        cname,
        choice
      )
    );
  }

  if ( iscale != 100 )
  {
    fscale = (GLfloat) ((double) iscale / 100.0);

    for ( i = 0; i < 3; i++ )
    {
      c[i] *= fscale;

      if      ( c[i] < 0.0 )
      {
        c[i] = 0.0;
      }
      else if ( c[i] > 1.0 )
      {
        c[i] = 1.0;
      }
    }
  }

  return c;
}


//-----------------------------------------------------------------------
//   getChoice
//-----------------------------------------------------------------------


String Color::getChoice ()
{
  StringBuffer  buf;

  buf.reserve ( 32 );

  for ( int i = 0; DEFS[i].name; i++ )
  {
    if ( i > 0 )
    {
      if ( DEFS[i + 1].name )
      {
        print ( buf, ", " );
      }
      else
      {
        print ( buf, " or " );
      }
    }

    print ( buf, '`', DEFS[i].name, '\'' );
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   findName
//-----------------------------------------------------------------------


bool Color::findName

  ( String&  name,
    double   eps ) const

{
  Color   col;

  double  diff, minDiff;
  double  delta;
  double  myScale;
  double  colScale;

  bool    found;
  int     level;
  int     i, j;


  if ( std::fabs( data_[3] - 1.0 ) > eps )
  {
    return false;
  }

  found   = false;
  minDiff = eps;

  for ( myScale = 0.0, j = 0; j < 3; j++ )
  {
    myScale += data_[j] * data_[j];
  }

  if ( myScale < 1.0e-6 )
  {
    myScale = 1.0;
  }
  else
  {
    myScale = 1.0 / std::sqrt ( myScale );
  }

  for ( i = 0; DEFS[i].name; i++ )
  {
    col = DEFS[i].color ();

    for ( colScale = 0.0, j = 0; j < 3; j++ )
    {
      colScale += col[j] * col[j];
    }

    if ( colScale < 1.0e-6 )
    {
      colScale = 1.0;
    }
    else
    {
      colScale = 1.0 / std::sqrt ( colScale );
    }

    for ( diff = 0.0, j = 0; j < 3; j++ )
    {
      delta = myScale * data_[j] - colScale * col[j];
      diff += delta * delta;
    }

    diff = std::sqrt ( diff );

    if ( diff <= minDiff )
    {
      minDiff = diff;
      found   = true;
      level   = (int) (0.5 + 100.0 * colScale / myScale);

      if ( level < 100 )
      {
        name = String::format ( "%s%d", DEFS[i].name, level );
      }
      else
      {
        name = DEFS[i].name;
      }
    }
  }

  return found;
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Color::toString () const
{
  String  s;

  double  r = data_[0];
  double  g = data_[1];
  double  b = data_[2];
  double  a = data_[3];


  if ( ! findName( s ) )
  {
    if ( std::fabs ( a - 1.0 ) < 0.01 )
    {
      s = String::format ( "[ %g, %g, %g ]", r, g, b );
    }
    else
    {
      s = String::format ( "[ %g, %g, %g, %g ]", r, g, b, a );
    }
  }

  return s;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


bool Color::configure

  ( const String&      name,
    const Properties&  props )

{
  using jem::util::Dictionary;

  Ref<Object>  obj;

  if ( ! props.find( obj, name ) )
  {
    return false;
  }

  if ( isInstance<String>( obj ) )
  {
    try
    {
      *this = parse ( toValue<String>( obj ) );
    }
    catch ( const IllegalInputException& ex )
    {
      props.propertyError ( name, ex.what() );
    }

    return true;
  }

  if ( isInstance< Array<double> >( obj ) )
  {
    Array<double>  tmp;

    props.get ( tmp, name, 0.0, 1.0 );

    if ( tmp.size() < 3 || tmp.size() > 4 )
    {
      props.propertyError (
        name,
        String::format (
          "invalid array length (%d); should be 3 or 4", tmp.size ()
        )
      );
    }

    data_[0] = (GLfloat) tmp[0];
    data_[1] = (GLfloat) tmp[1];
    data_[2] = (GLfloat) tmp[2];

    if ( tmp.size() < 4 )
    {
      data_[3] = 1.0F;
    }
    else
    {
      data_[3] = (GLfloat) tmp[3];
    }

    return true;
  }

  if ( isInstance<Dictionary>( obj ) )
  {
    Properties  tmp = props.getProps ( name );

    double      r, g, b, a;

    a = 1.0;

    tmp.get  ( r, "r", 0.0, 1.0 );
    tmp.get  ( g, "g", 0.0, 1.0 );
    tmp.get  ( b, "b", 0.0, 1.0 );
    tmp.find ( a, "a", 0.0, 1.0 );

    data_[0] = (GLfloat) r;
    data_[1] = (GLfloat) g;
    data_[2] = (GLfloat) b;
    data_[3] = (GLfloat) a;

    return true;
  }

  props.propertyError (
    name,
    "property has wrong type; expected a color"
  );

  return false;
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Color::getConfig

  ( const String&      name,
    const Properties&  conf ) const

{
  String  s;

  if ( findName( s ) )
  {
    conf.set ( name, s );
  }
  else
  {
    Array<double>  tmp;

    int            n;

    if ( std::fabs( data_[3] - 1.0 ) < 0.01 )
    {
      n = 3;
    }
    else
    {
      n = 4;
    }

    tmp.resize ( n );

    for ( int i = 0; i < n; i++ )
    {
      tmp[i] = data_[i];
    }

    conf.set ( name, tmp );
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   input operator
//-----------------------------------------------------------------------


void            read

  ( TextInput&    in,
    Color&        col )

{
  Color   tmp;

  double  x;
  int     ch;
  int     i;


  in.skipWhite ();
  ch = in.read ();

  if ( ch < 0 )
  {
    throw ParseException ( JEM_FUNC, "unexpected end of input" );
  }

  if ( ch != '[' )
  {
    String  s;

    in.pushBack ( ch );

    read ( in, s );

    try
    {
      col = Color::byName ( s );
    }
    catch ( const IllegalInputException& ex )
    {
      throw ParseException ( JEM_FUNC, ex.what() );
    }

    return;
  }

  tmp[3] = 1.0;

  for ( i = 0; i < 4; i++ )
  {
    in.skipWhite ();
    ch = in.read ();

    if ( ch < 0 )
    {
      throw ParseException ( JEM_FUNC, "unexpected end of input" );
    }

    if ( ch == ']' )
    {
      break;
    }

    if ( ch == ',' )
    {
      throw ParseException ( JEM_FUNC,
                             "missing color component" );
    }

    in.pushBack ( ch );

    read ( in, x );

    if ( x < 0.0 || x > 1.0 )
    {
      throw ParseException (
        JEM_FUNC,
        "color component out of range; valid range is [0,1]"
      );
    }

    tmp[i] = (GLfloat) x;

    in.skipWhite ();
    ch = in.read ();

    if ( ch < 0 )
    {
      throw ParseException ( JEM_FUNC, "unexpected end of input" );
    }

    if ( ch == ']' )
    {
      break;
    }

    if ( ch != ',' )
    {
      throw ParseException (
        JEM_FUNC,
        String::format ( "parse error before `%c\'", ch )
      );
    }
  }

  if ( i < 2 )
  {
    throw ParseException (
      JEM_FUNC,
      "incomplete color vector; "
      "at least three components are required"
    );
  }

  if ( ch != ']' )
  {
    throw ParseException (
      JEM_FUNC,
      "invalid color vector: missing `]\'"
    );
  }

  col = tmp;
}


//-----------------------------------------------------------------------
//   clamp
//-----------------------------------------------------------------------


bool clamp ( Color& c ) noexcept
{
  bool  r = false;

  for ( int i = 0; i < 4; i++ )
  {
    if      ( c[i] < 0.0 )
    {
      c[i] = 0.0;
      r    = true;
    }
    else if ( c[i] > 1.0 )
    {
      c[i] = 1.0;
      r    = true;
    }
  }

  return r;
}


JEM_END_PACKAGE( gl )
