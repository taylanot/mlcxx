
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

#ifndef JEM_BASE_ASSERT_H
#define JEM_BASE_ASSERT_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


inline bool       checkIndex

  ( short           index,
    int             ubound );

inline bool       checkIndex

  ( unsigned short  index,
    int             ubound );

inline bool       checkIndex

  ( int             index,
    int             ubound );

inline bool       checkIndex

  ( unsigned int    index,
    int             ubound );

inline bool       checkIndex

  ( lint            index,
    int             ubound );

inline bool       checkIndex

  ( ulint           index,
    int             ubound );

inline bool       checkIndex

  ( short           index,
    lint            ubound );

inline bool       checkIndex

  ( unsigned short  index,
    lint            ubound );

inline bool       checkIndex

  ( int             index,
    lint            ubound );

inline bool       checkIndex

  ( unsigned int    index,
    lint            ubound );

inline bool       checkIndex

  ( lint            index,
    lint            ubound );

inline bool       checkIndex

  ( ulint           index,
    lint            ubound );


//-----------------------------------------------------------------------
//   private helper functions
//-----------------------------------------------------------------------


void              assertFailed_

  ( const char*     expr,
    const char*     mesg,
    const char*     func,
    const char*     file,
    int             line );

void              assertNoThrowFailed_

  ( const char*     expr,
    const char*     mesg,
    const char*     func,
    const char*     file,
    int             line );

void              precheckFailed_

  ( const char*     expr,
    const char*     mesg,
    const char*     func,
    const char*     file,
    int             line );


JEM_END_PACKAGE_BASE


//-----------------------------------------------------------------------
//   macro JEM_ASSERT
//-----------------------------------------------------------------------


#undef JEM_ASSERT

#ifdef NDEBUG

#  define JEM_ASSERT( expr )  ((void) 0)

#else

#  define JEM_ASSERT( expr )                            \
                                                        \
  if ( ! (expr) )                                       \
    ::jem::assertFailed_ ( #expr, 0, JEM_FUNC,          \
                           __FILE__, (int) __LINE__ )

#endif


//-----------------------------------------------------------------------
//   macro JEM_ASSERT2
//-----------------------------------------------------------------------


#undef JEM_ASSERT2

#ifdef NDEBUG

#  define JEM_ASSERT2( expr, errmsg )  ((void) 0)

#else

#  define JEM_ASSERT2( expr, errmsg )                   \
                                                        \
  if ( ! (expr) )                                       \
    ::jem::assertFailed_ ( #expr, errmsg, JEM_FUNC,     \
                           __FILE__, (int) __LINE__ )

#endif


//-----------------------------------------------------------------------
//   macro JEM_ASSERT_NOTHROW
//-----------------------------------------------------------------------


#undef JEM_ASSERT_NOTHROW

#ifdef NDEBUG

#  define JEM_ASSERT_NOTHROW( expr )  ((void) 0)

#else

#  define JEM_ASSERT_NOTHROW( expr )                            \
                                                                \
  if ( ! (expr) )                                               \
    ::jem::assertNoThrowFailed_ ( #expr, 0, JEM_FUNC,           \
                                  __FILE__, (int) __LINE__ )

#endif


//-----------------------------------------------------------------------
//   macro JEM_ASSERT2_NOTHROW
//-----------------------------------------------------------------------


#undef JEM_ASSERT2_NOTHROW

#ifdef NDEBUG

#  define JEM_ASSERT2_NOTHROW( expr, errmsg )  ((void) 0)

#else

#  define JEM_ASSERT2_NOTHROW( expr, errmsg )                   \
                                                                \
  if ( ! (expr) )                                               \
    ::jem::assertNoThrowFailed_ ( #expr, errmsg, JEM_FUNC,      \
                                  __FILE__, (int) __LINE__ )

#endif

//-----------------------------------------------------------------------
//   macro JEM_PRECHECK
//-----------------------------------------------------------------------


#undef  JEM_PRECHECK
#define JEM_PRECHECK( expr )                            \
                                                        \
  if ( ! (expr) )                                       \
    ::jem::precheckFailed_ ( #expr, 0, JEM_FUNC,        \
                             __FILE__, (int) __LINE__ )


//-----------------------------------------------------------------------
//   macro JEM_PRECHECK2
//-----------------------------------------------------------------------


#undef  JEM_PRECHECK2
#define JEM_PRECHECK2( expr, errmsg )                   \
                                                        \
  if ( ! (expr) )                                       \
    ::jem::precheckFailed_ ( #expr, errmsg, JEM_FUNC,   \
                             __FILE__, (int) __LINE__ )


JEM_BEGIN_PACKAGE_BASE


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   checkIndex
//-----------------------------------------------------------------------


inline bool       checkIndex

  ( short           index,
    int             ubound )

{
  return (index >= 0 && index < ubound);
}


inline bool       checkIndex

  ( unsigned short  index,
    int             ubound )

{
  return (index < (unsigned int) ubound);
}


inline bool       checkIndex

  ( int             index,
    int             ubound )

{
  return (index >= 0 && index < ubound);
}


inline bool       checkIndex

  ( unsigned int    index,
    int             ubound )

{
  return (index < (unsigned int) ubound);
}


inline bool       checkIndex

  ( lint            index,
    int             ubound )

{
  return (index >= 0 && index < ubound);
}


inline bool       checkIndex

  ( ulint           index,
    int             ubound )

{
  return (index < (ulint) ubound);
}


inline bool       checkIndex

  ( short           index,
    lint            ubound )

{
  return (index >= 0 && index < ubound);
}


inline bool       checkIndex

  ( unsigned short  index,
    lint            ubound )

{
  return (index < (ulint) ubound);
}


inline bool       checkIndex

  ( int             index,
    lint            ubound )

{
  return (index >= 0 && index < ubound);
}


inline bool       checkIndex

  ( unsigned int    index,
    lint            ubound )

{
  return (index < (ulint) ubound);
}


inline bool       checkIndex

  ( lint            index,
    lint            ubound )

{
  return (index >= 0_lint && index < ubound);
}


inline bool       checkIndex

  ( ulint           index,
    lint            ubound )

{
  return (index < (ulint) ubound);
}


JEM_END_PACKAGE_BASE

#endif
