
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

#ifndef JEM_IO_NUMBERFORMAT_H
#define JEM_IO_NUMBERFORMAT_H

#include <jem/base/String.h>


JEM_BEGIN_PACKAGE( io )


class DataInput;
class DataOtuput;
class PrintWriter;


//-----------------------------------------------------------------------
//   class NumberFormat
//-----------------------------------------------------------------------


class NumberFormat
{
 public:

  friend class          PrintWriter;

  static const int      MAX_WIDTH   = 32;
  static const int      MAX_DIGITS  = 32;
  static const int      MIN_BUFSIZE = MAX_WIDTH + MAX_DIGITS + 16;

  static const int      SCIENTIFIC;
  static const int      UPPERCASE;
  static const int      SHOW_SIGN;


  inline                NumberFormat      ()       noexcept;

  void                  setDefaults       ()       noexcept;

  void                  setFlag

    ( int                 flag,
      bool                choice = true );

  void                  setFlags

    ( int                 flags );

  inline void           setScientific

    ( bool                choice = true );

  inline void           setUppercase

    ( bool                choice = true );

  inline void           setShowSign

    ( bool                choice = true );

  void                  setIntegerBase

    ( int                 base  );

  void                  setIntegerWidth

    ( int                 n );

  void                  setIntegerDigits

    ( int                 n );

  void                  setFloatWidth

    ( int                 n );

  void                  setFractionDigits

    ( int                 n );

  inline int            getFlags          () const noexcept;
  inline int            getIntegerBase    () const noexcept;
  inline int            getIntegerWidth   () const noexcept;
  inline int            getIntegerDigits  () const noexcept;
  inline int            getFloatWidth     () const noexcept;
  inline int            getFractionDigits () const noexcept;

  int                   print

    ( char*               buf,
      lint                number )           const;

  String                print

    ( lint                number )           const;

  int                   print

    ( char*               buf,
      double              number )           const;

  String                print

    ( double              number )           const;


 private:

  void                  updateFormat_     ()       noexcept;


 private:

  class                 Utils_;

  static const char*    LOWER_DIGITS_;
  static const char*    UPPER_DIGITS_;


  int                   flags_;
  int                   intBase_;
  int                   intWidth_;
  int                   intDigits_;
  int                   floatWidth_;
  int                   fracDigits_;

  char                  format_[8];

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


void                    decode

  ( DataInput&            in,
    NumberFormat&         nf );

void                    encode

  ( DataOutput&           out,
    const NumberFormat&   nf );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class NumberFormat
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NumberFormat::NumberFormat () noexcept
{
  setDefaults ();
}


//-----------------------------------------------------------------------
//   setScientific
//-----------------------------------------------------------------------


inline void NumberFormat::setScientific ( bool choice )
{
  setFlag ( SCIENTIFIC, choice );
}


//-----------------------------------------------------------------------
//   setUppercase
//-----------------------------------------------------------------------


inline void NumberFormat::setUppercase ( bool choice )
{
  setFlag ( UPPERCASE, choice );
}


//-----------------------------------------------------------------------
//   setShowSign
//-----------------------------------------------------------------------


inline void NumberFormat::setShowSign ( bool choice )
{
  setFlag ( SHOW_SIGN, choice );
}


//-----------------------------------------------------------------------
//   getFlags
//-----------------------------------------------------------------------


inline int NumberFormat::getFlags () const noexcept
{
  return flags_;
}


//-----------------------------------------------------------------------
//   getIntegerBase
//-----------------------------------------------------------------------


inline int NumberFormat::getIntegerBase () const noexcept
{
  return intBase_;
}


//-----------------------------------------------------------------------
//   getIntegerWidth
//-----------------------------------------------------------------------


inline int NumberFormat::getIntegerWidth () const noexcept
{
  return intWidth_;
}


//-----------------------------------------------------------------------
//   getIntegerDigits
//-----------------------------------------------------------------------


inline int NumberFormat::getIntegerDigits () const noexcept
{
  return intDigits_;
}


//-----------------------------------------------------------------------
//   getFloatWidth
//-----------------------------------------------------------------------


inline int NumberFormat::getFloatWidth () const noexcept
{
  return floatWidth_;
}


//-----------------------------------------------------------------------
//   getFractionDigits
//-----------------------------------------------------------------------


inline int NumberFormat::getFractionDigits () const noexcept
{
  return fracDigits_;
}


JEM_END_PACKAGE( io )

#endif
