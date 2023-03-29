
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

#ifndef JEM_CCIDENT_H
#define JEM_CCIDENT_H

#if   defined(__INTEL_COMPILER)

#  define JEM_CXX_ICC
#  define JEM_CXX_NAME     "icc"
#  define JEM_CXX_VERSION  __INTEL_COMPILER

#elif defined(__GNUC__)

#  define JEM_CXX_GCC
#  define JEM_CXX_NAME     "gcc"

#  if defined(__GNUC_PATCHLEVEL__)
#    define JEM_CXX_VERSION (__GNUC__ * 10000 + \
                             __GNUC_MINOR__ * 100 + \
                             __GNUC_PATCHLEVEL__)
#  else
#    define JEM_CXX_VERSION (__GNUC__ * 10000 + \
                             __GNUC_MINOR__ * 100)
#  endif

#  if defined(__clang__)
#    define JEM_CXX_NAME2  "clang"
#  endif

#elif defined(_MSC_VER)

#  define JEM_CXX_MSC
#  define JEM_CXX_NAME     "msc"
#  define JEM_CXX_VERSION  _MSC_VER

#else

#  undef  JEM_CXX_NAME
#  undef  JEM_CXX_VERSION

#endif

#endif
