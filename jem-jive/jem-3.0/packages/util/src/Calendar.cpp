
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


#include <ctime>
#include <climits>
#include <cstring>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/SystemException.h>
#include <jem/base/native/Time.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Calendar.h>
#include <jem/util/DateException.h>


JEM_DEFINE_SERIABLE_CLASS ( jem::util::Calendar );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class Calendar
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Calendar::FIELD_ERROR_ = "invalid Calendar field";

const int    Calendar::MAXIMA_[FIELD_COUNT] =
{
  61, 59, 23, 6, 31, 365, 11, INT_MAX
};

const int    Calendar::MINIMA_[FIELD_COUNT] =
{
  0, 0, 0, 0, 1, 0, 0, INT_MIN
};

const int    Calendar::SMALLEST_MAXIMA_[FIELD_COUNT] =
{
  59, 59, 23, 6, 28, 364, 11, INT_MAX
};

const int    Calendar::GREATEST_MINIMA_[FIELD_COUNT] =
{
  0, 0, 0, 0, 1, 0, 0, INT_MIN
};


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Calendar::Calendar () :

  isLenient_ ( true  ),
  isDirty_   ( false )

{
  setTime ( Time::now() );
}


Calendar::Calendar ( const Time& t ) :

  isLenient_ ( true  ),
  isDirty_   ( false )

{
  setTime ( t );
}


Calendar::Calendar ( const Calendar& rhs )
{
  for ( int i = 0; i < FIELD_COUNT; i++ )
  {
    fields_[i] = rhs.fields_[i];
    isSet_ [i] = rhs.isSet_ [i];
  }

  isLenient_ = rhs.isLenient_;
  isDirty_   = rhs.isDirty_;
}


Calendar::~Calendar ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void Calendar::readFrom ( io::ObjectInput& in )
{
  in.decode ( fields_, FIELD_COUNT );
  in.decode ( isSet_,  FIELD_COUNT );

  decode ( in, isLenient_, isDirty_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void Calendar::writeTo ( io::ObjectOutput& out ) const
{
  out.encode  ( fields_, FIELD_COUNT );
  out.encode  ( isSet_,  FIELD_COUNT );

  encode ( out, isLenient_, isDirty_ );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String Calendar::toString () const
{
  String str;

  try
  {
    str = getTime().toString ();
  }
  catch ( const DateException& )
  {
    str = Super::toString ();
  }

  return str;
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint Calendar::hashValue () const
{
  lint value;

  try
  {
    value = TypeTraits<Time>::hashValue ( getTime() );
  }
  catch ( const DateException& )
  {
    value = Super::hashValue ();
  }

  return value;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool Calendar::equals ( const Ref<Object>&  obj ) const
{
  Calendar*  cal    = dynamicCast<Calendar*> ( obj );
  bool       result = false;

  if ( cal )
  {
    try
    {
      result = (getTime() == cal->getTime());
    }
    catch ( const DateException& )
    {
      result = (this == cal);
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> Calendar::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   getTime
//-----------------------------------------------------------------------


Time Calendar::getTime () const
{
  time_t     t;
  struct tm  date;

  std::memset ( &date, 0x0, sizeof(date) );

  date.tm_sec   = fields_[SECOND];
  date.tm_min   = fields_[MINUTE];
  date.tm_hour  = fields_[HOUR];
  date.tm_mday  = fields_[DAY_OF_MONTH];
  date.tm_mon   = fields_[MONTH];
  date.tm_year  = fields_[YEAR] - 1900;
  date.tm_isdst = -1;

  t = ::mktime ( &date );

  if ( t == (time_t) -1 )
  {
    throw DateException ( JEM_FUNC, "illegal date specified" );
  }

  return Time ( (lint) t );
}


//-----------------------------------------------------------------------
//   setTime
//-----------------------------------------------------------------------


void Calendar::setTime ( const Time& t )
{
  computeFields_ ( fields_, t );

  isDirty_ = false;

  for ( int i = 0; i < FIELD_COUNT; i++ )
  {
    isSet_[i] = false;
  }
}


//-----------------------------------------------------------------------
//   getField
//-----------------------------------------------------------------------


int Calendar::getField ( int field ) const
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  if ( isDirty_ )
  {
    Calendar*  self = const_cast<Calendar*> ( this );

    computeFields_ ( self->fields_, getTime() );

    self->isDirty_ = false;
  }

  return fields_[field];
}


//-----------------------------------------------------------------------
//   addToField
//-----------------------------------------------------------------------


void Calendar::addToField ( int field, int amount )
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  setField_ ( field, fields_[field] + amount );

  if ( ! isLenient_ )
  {
    checkFields_ ();
  }
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void Calendar::setField ( int field, int value )
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  setField_ ( field, value );

  if ( ! isLenient_ )
  {
    checkFields_ ();
  }
}


//-----------------------------------------------------------------------
//   setDate
//-----------------------------------------------------------------------


void Calendar::setDate ( int year, int month, int mday )
{
  setField_ ( YEAR,         year  );
  setField_ ( MONTH,        month );
  setField_ ( DAY_OF_MONTH, mday  );

  if ( ! isLenient_ )
  {
    checkFields_ ();
  }
}


void Calendar::setDate ( int year, int month, int mday,
                         int hour, int min )
{
  setField_ ( YEAR,         year  );
  setField_ ( MONTH,        month );
  setField_ ( DAY_OF_MONTH, mday  );
  setField_ ( HOUR,         hour  );
  setField_ ( MINUTE,       min   );

  if ( ! isLenient_ )
  {
    checkFields_ ();
  }
}


void Calendar::setDate ( int year, int month, int mday,
                         int hour, int min,   int sec )
{
  setField_ ( YEAR,         year  );
  setField_ ( MONTH,        month );
  setField_ ( DAY_OF_MONTH, mday  );
  setField_ ( HOUR,         hour  );
  setField_ ( MINUTE,       min   );
  setField_ ( SECOND,       sec   );

  if ( ! isLenient_ )
  {
    checkFields_ ();
  }
}


//-----------------------------------------------------------------------
//   getMaximum & getMinimum
//-----------------------------------------------------------------------


int Calendar::getMaximum ( int field )
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  return MAXIMA_[field];
}


int Calendar::getMinimum ( int field )
{
  JEM_PRECHECK ( field >= 0 && field < FIELD_COUNT );

  return MINIMA_[field];
}


//-----------------------------------------------------------------------
//   getSmallestMaximum & getGreatestMinimum
//-----------------------------------------------------------------------


int Calendar::getSmallestMaximum ( int field )
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  return SMALLEST_MAXIMA_[field];
}


int Calendar::getGreatestMinimum ( int field )
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  return GREATEST_MINIMA_[field];
}


//-----------------------------------------------------------------------
//   getActualMaximum
//-----------------------------------------------------------------------


int Calendar::getActualMaximum ( int field ) const
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  int  n = 0;

  if ( MAXIMA_[field] == SMALLEST_MAXIMA_[field] )
  {
    return MAXIMA_[field];
  }
  else
  {
    Ref<Calendar>  c = staticCast<Calendar> ( clone() );

    n = SMALLEST_MAXIMA_[field] + 1;

    while ( n < MAXIMA_[field] )
    {
      c->setField ( field, n );

      if ( c->getField( field ) != n )
      {
        n--;
        break;
      }
      n++;
    }

    return n;
  }
}


//-----------------------------------------------------------------------
//   getActualMinimum
//-----------------------------------------------------------------------


int Calendar::getActualMinimum ( int field ) const
{
  JEM_PRECHECK2 ( field >= 0 && field < FIELD_COUNT, FIELD_ERROR_ );

  int  n = 0;

  if ( MINIMA_[field] == GREATEST_MINIMA_[field] )
  {
    return MINIMA_[field];
  }
  else
  {
    Ref<Calendar>  c = staticCast<Calendar> ( clone() );

    n = GREATEST_MINIMA_[field] - 1;

    while ( n > MINIMA_[field] )
    {
      c->setField ( field, n );

      if ( c->getField( field ) != n )
      {
        n++;
        break;
      }
      n--;
    }

    return n;
  }
}


//-----------------------------------------------------------------------
//   setLenient
//-----------------------------------------------------------------------


void Calendar::setLenient ( bool flag )
{
  if ( flag != isLenient_ )
  {
    isLenient_ = flag;

    if ( ! isLenient_ )
    {
      checkFields_ ();
    }
  }
}


//-----------------------------------------------------------------------
//   isLenient
//-----------------------------------------------------------------------


bool Calendar::isLenient () const noexcept
{
  return isLenient_;
}


//-----------------------------------------------------------------------
//   setField_
//-----------------------------------------------------------------------


void Calendar::setField_ ( int field, int value )
{
  fields_[field] = value;
  isSet_[field]  = true;
  isDirty_       = true;
}


//-----------------------------------------------------------------------
//   checkFields_
//-----------------------------------------------------------------------


void Calendar::checkFields_ () const
{
  int   fields[FIELD_COUNT];
  bool  valid;

  computeFields_ ( fields, getTime() );

  valid = true;

  for ( int i = 0; i < FIELD_COUNT; i++ )
  {
    if ( isSet_[i] )
    {
      valid = valid && (fields[i] == fields_[i]);
    }
  }

  if ( ! valid )
  {
    throw DateException( JEM_FUNC, "invalid date" );
  }
}


//-----------------------------------------------------------------------
//   computeFields_
//-----------------------------------------------------------------------


void Calendar::computeFields_ ( int* fields, const Time& t )
{
  struct tm  date;

  NativeTime::getLocalTime ( date, t );

  fields[SECOND]       = date.tm_sec;
  fields[MINUTE]       = date.tm_min;
  fields[HOUR]         = date.tm_hour;
  fields[DAY_OF_WEEK]  = date.tm_wday;
  fields[DAY_OF_MONTH] = date.tm_mday;
  fields[DAY_OF_YEAR]  = date.tm_yday;
  fields[MONTH]        = date.tm_mon;
  fields[YEAR]         = date.tm_year + 1900;
}


JEM_END_PACKAGE( util )
