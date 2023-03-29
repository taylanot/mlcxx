
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

#ifndef JEM_UTIL_CALENDAR_H
#define JEM_UTIL_CALENDAR_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jem/base/Time.h>
#include <jem/io/Serializable.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Calendar
//-----------------------------------------------------------------------


class Calendar : public Object,
                 public Clonable,
                 public io::Serializable
{
 public:

  JEM_DECLARE_CLASS   ( Calendar, Object );

  enum                  Fields
  {
                          SECOND = 0,
                          MINUTE,
                          HOUR,
                          DAY_OF_WEEK,
                          DAY_OF_MONTH,
                          DAY_OF_YEAR,
                          MONTH,
                          YEAR,
                          FIELD_COUNT
  };

  enum                  Months
  {
                          JANUARY = 0,
                          FEBRUARI,
                          MARCH,
                          APRIL,
                          MAY,
                          JUNE,
                          JULY,
                          AUGUST,
                          SEPTEMBER,
                          OCTOBER,
                          NOVEMBER,
                          DECEMBER
  };

  enum                  WeekDays
  {
                          SUNDAY = 0,
                          MONDAY,
                          TUESDAY,
                          WEDNESDAY,
                          THURSDAY,
                          FRIDAY,
                          SATURDAY
  };


                        Calendar  ();

  explicit              Calendar

    ( const Time&         t );

                        Calendar

    ( const Calendar&     rhs );

  virtual void          readFrom

    ( io::ObjectInput&    in  )                override;

  virtual void          writeTo

    ( io::ObjectOutput&   out )          const override;

  virtual String        toString      () const override;
  virtual lint          hashValue     () const override;

  virtual bool          equals

    ( const Ref<Object>&  rhs )          const override;

  virtual Ref<Object>   clone         () const override;

  Time                  getTime       () const;

  void                  setTime

    ( const Time&         t );

  int                   getField

    ( int                 field )        const;

  void                  addToField

    ( int                 field,
      int                 amount );

  void                  setField

    ( int                 field,
      int                 value );

  void                  setDate

    ( int                 year,
      int                 month,
      int                 mday );

  void                  setDate

    ( int                 year,
      int                 month,
      int                 mday,
      int                 hour,
      int                 min );

  void                  setDate

    ( int                 year,
      int                 month,
      int                 mday,
      int                 hour,
      int                 min,
      int                 sec );

  static int            getMaximum

    ( int                 field );

  static int            getMinimum

    ( int                 field );

  static int            getSmallestMaximum

    ( int                 field );

  static int            getGreatestMinimum

    ( int                 field );

  int                   getActualMaximum

    ( int                 field )        const;

  int                   getActualMinimum

    ( int                 field )        const;

  void                  setLenient

    ( bool                flag  );

  bool                  isLenient     () const noexcept;


 protected:

  virtual              ~Calendar      ();


 private:

  void                  setField_

    ( int                 field,
      int                 value );

  void                  checkFields_  () const;

  static void           computeFields_

    ( int*                fields,
      const Time&         t );


 private:

  static const char*    FIELD_ERROR_;

  static const int      MAXIMA_          [FIELD_COUNT];
  static const int      MINIMA_          [FIELD_COUNT];
  static const int      SMALLEST_MAXIMA_ [FIELD_COUNT];
  static const int      GREATEST_MINIMA_ [FIELD_COUNT];

  int                   fields_          [FIELD_COUNT];
  bool                  isSet_           [FIELD_COUNT];

  bool                  isLenient_;
  bool                  isDirty_;

};


JEM_END_PACKAGE( util )

#endif
