
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

#ifndef JEM_PRAGMAS_H
#define JEM_PRAGMAS_H


//-----------------------------------------------------------------------
//   JEM_PRAGMA
//-----------------------------------------------------------------------


#undef JEM_PRAGMA

#if   defined(__INTEL_COMPILER)

#  if defined(__ia64) && (__INTEL_COMPILER >= 810)
#    define JEM_PRAGMA( p ) _Pragma( #p )
#  endif

#elif defined(__GNUC__)

#  define JEM_PRAGMA( p ) _Pragma( #p )

#endif

#ifndef JEM_PRAGMA
#  define JEM_PRAGMA( p )
#endif


//-----------------------------------------------------------------------
//   JEM_IVDEP
//-----------------------------------------------------------------------


#undef JEM_IVDEP

#if defined(__INTEL_COMPILER) && defined(__ia64)
#  define JEM_IVDEP JEM_PRAGMA( ivdep )
#endif

#ifndef JEM_IVDEP
#  define JEM_IVDEP
#endif


//-----------------------------------------------------------------------
//   JEM_NOPREFETCH
//-----------------------------------------------------------------------


#undef JEM_NOPREFETCH

#if defined(__INTEL_COMPILER) && defined(__ia64)
#  define JEM_NOPREFETCH( a ) JEM_PRAGMA( noprefetch a )
#endif

#ifndef JEM_NOPREFETCH
#  define JEM_NOPREFETCH( a )
#endif


//-----------------------------------------------------------------------
//   JEM_PREFETCH_NONE
//-----------------------------------------------------------------------


#undef JEM_PREFETCH_NONE

#if defined(__INTEL_COMPILER) && defined(__ia64)
#  define JEM_PREFETCH_NONE JEM_PRAGMA( noprefetch )
#endif

#ifndef JEM_PREFETCH_NONE
#  define JEM_PREFETCH_NONE
#endif


#endif
