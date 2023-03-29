
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


#include <cstdio>
#include <cstring>
#include <jem/io/DataInput.h>
#include <jem/io/DataOutput.h>
#include <jem/io/NumberFormat.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class NumberFormat::Utils_
//=======================================================================


class NumberFormat::Utils_
{
 public:

  static int  clamp ( int n, int lbound, int ubound )
  {
    if      ( n < lbound )
    {
      return lbound;
    }
    else if ( n > ubound )
    {
      return ubound;
    }
    else
    {
      return n;
    }
  }

};


//=======================================================================
//   class NumberFormat
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int    NumberFormat::SCIENTIFIC    = 1 << 0;
const int    NumberFormat::UPPERCASE     = 1 << 1;
const int    NumberFormat::SHOW_SIGN     = 1 << 2;

const char*  NumberFormat::LOWER_DIGITS_ = "0123456789abcdefx";
const char*  NumberFormat::UPPER_DIGITS_ = "0123456789ABCDEFX";


//-----------------------------------------------------------------------
//   setDefaults
//-----------------------------------------------------------------------


void NumberFormat::setDefaults () noexcept
{
  flags_       = 0;
  intBase_     = 10;
  intWidth_    = 0;
  intDigits_   = 0;
  floatWidth_  = 0;
  fracDigits_  = 6;

  updateFormat_ ();
}


//-----------------------------------------------------------------------
//   setFlag
//-----------------------------------------------------------------------


void NumberFormat::setFlag

  ( int   flag,
    bool  choice )

{
  if ( choice )
  {
    if ( (flags_ & flag) != flag )
    {
      flags_ |= flag;

      updateFormat_ ();
    }
  }
  else
  {
    if ( flags_ & flag )
    {
      flags_ &= ~flag;

      updateFormat_ ();
    }
  }
}


//-----------------------------------------------------------------------
//   setFlags
//-----------------------------------------------------------------------


void NumberFormat::setFlags ( int flags )
{
  if ( flags_ != flags )
  {
    flags_ = flags;

    updateFormat_ ();
  }
}


//-----------------------------------------------------------------------
//   setIntegerBase
//-----------------------------------------------------------------------


void NumberFormat::setIntegerBase ( int n )
{
  intBase_ = Utils_::clamp ( n, 2, 16 );
}


//-----------------------------------------------------------------------
//   setIntegerWidth
//-----------------------------------------------------------------------


void NumberFormat::setIntegerWidth ( int n )
{
  intWidth_ = Utils_::clamp ( n, -MAX_WIDTH, MAX_WIDTH );
}


//-----------------------------------------------------------------------
//   setIntegerDigits
//-----------------------------------------------------------------------


void NumberFormat::setIntegerDigits ( int n )
{
  intDigits_ = Utils_::clamp ( n, 0, MAX_DIGITS );
}


//-----------------------------------------------------------------------
//   setFloatWidth
//-----------------------------------------------------------------------


void NumberFormat::setFloatWidth ( int n )
{
  floatWidth_ = Utils_::clamp ( n, -MAX_WIDTH, MAX_WIDTH );
}


//-----------------------------------------------------------------------
//   setFractionDigits
//-----------------------------------------------------------------------


void NumberFormat::setFractionDigits ( int n )
{
  fracDigits_ = Utils_::clamp ( n, 0, MAX_DIGITS );
}


//-----------------------------------------------------------------------
//   print ( long )
//-----------------------------------------------------------------------


int NumberFormat::print ( char*  buf, lint num ) const
{
  const int    N    = 8 * (int) sizeof(num);
  const int    prec = intDigits_;
  const lint   base = intBase_;

  const char*  digits;
  char         str[MIN_BUFSIZE];
  lint         powers[N];
  lint         p, n;
  int          i, j, k;


  i = 0;
  k = 0;
  n = num / base;

  if ( flags_ & UPPERCASE )
  {
    digits = UPPER_DIGITS_;
  }
  else
  {
    digits = LOWER_DIGITS_;
  }

  if ( num < 0 )
  {
    str[i++] = '-';

    if ( base == 16 )
    {
      str[i++] = '0';
      str[i++] = digits[16];
    }

    for ( p = -1; p >= n; p *= base )
    {
      powers[k++] = p;
    }

    powers[k++] = p;

    for ( j = k; j < prec; j++ )
    {
      str[i++] = '0';
    }

    for ( k--; k >= 0; k-- )
    {
      p = powers[k];
      j = 0;

      while ( num - p <= 0 )
      {
        j++;
        num -= p;
      }

      str[i++] = digits[j];
    }
  }
  else
  {
    if ( flags_ & SHOW_SIGN )
    {
      str[i++] = '+';
    }

    if ( base == 16 )
    {
      str[i++] = '0';
      str[i++] = digits[16];
    }

    for ( p = 1; p <= n; p *= base )
    {
      powers[k++] = p;
    }

    powers[k++] = p;

    for ( j = k; j < prec; j++ )
    {
      str[i++] = '0';
    }

    for ( k--; k >= 0; k-- )
    {
      p = powers[k];
      j = 0;

      while ( num - p >= 0 )
      {
        j++;
        num -= p;
      }

      str[i++] = digits[j];
    }
  }

  if ( i < intWidth_ )
  {
    k = intWidth_ - i;

    for ( j = 0; j < k; j++ )
    {
      buf[j] = ' ';
    }

    std::memcpy ( buf + j, str, (size_t) i );

    i += k;
  }
  else
  {
    std::memcpy ( buf, str, (size_t) i );

    k = -intWidth_;

    while ( i < k )
    {
      buf[i++] = ' ';
    }
  }

  return i;
}


String NumberFormat::print ( lint num ) const
{
  char  buf[MIN_BUFSIZE];

  int   n = print ( buf, num );

  return String ( buf, buf + n );
}


//-----------------------------------------------------------------------
//   print ( double )
//-----------------------------------------------------------------------


int NumberFormat::print ( char* buf, double num ) const
{
  return std::sprintf ( buf, format_, floatWidth_, fracDigits_, num );
}


String NumberFormat::print ( double num ) const
{
  char  buf[MIN_BUFSIZE];

  int   n = print ( buf, num );

  return String ( buf, n );
}


//-----------------------------------------------------------------------
//   updateFormat_
//-----------------------------------------------------------------------


void NumberFormat::updateFormat_ () noexcept
{
  int  i = 0;

  format_[i++]   = '%';
  format_[i++]   = '#';

  if ( flags_ & SHOW_SIGN )
  {
    format_[i++] = '+';
  }

  format_[i++]   = '*';
  format_[i++]   = '.';
  format_[i++]   = '*';

  if ( flags_ & UPPERCASE )
  {
    format_[i++] = (flags_ & SCIENTIFIC) ? 'E' : 'G';
  }
  else
  {
    format_[i++] = (flags_ & SCIENTIFIC) ? 'e' : 'g';
  }

  format_[i++]   = '\0';
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


void                    decode

  ( DataInput&            in,
    NumberFormat&         nf )

{
  int  buf[6];

  in.decode ( buf, 6 );

  nf.setFlags          ( buf[0] );
  nf.setIntegerBase    ( buf[1] );
  nf.setIntegerWidth   ( buf[2] );
  nf.setIntegerDigits  ( buf[3] );
  nf.setFloatWidth     ( buf[4] );
  nf.setFractionDigits ( buf[5] );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


void                    encode

  ( DataOutput&           out,
    const NumberFormat&   nf )
{
  int  buf[6];

  buf[0] = nf.getFlags          ();
  buf[1] = nf.getIntegerBase    ();
  buf[2] = nf.getIntegerWidth   ();
  buf[3] = nf.getIntegerDigits  ();
  buf[4] = nf.getFloatWidth     ();
  buf[5] = nf.getFractionDigits ();

  out.encode ( buf, 6 );
}


JEM_END_PACKAGE( io )
