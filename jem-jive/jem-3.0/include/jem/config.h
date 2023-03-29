
// Generated on Wed Dec 15 18:05:32 CET 2021.
// CHANGES TO THIS FILE WILL BE LOST.

#ifndef JEM_CONFIG_H
#define JEM_CONFIG_H

#define JEM_OS_LINUX
#define JEM_ARCH_X86_64

#define JEM_USE_THREADS
#define JEM_OS_POSIX
#define JEM_OS_UNIX98

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


#ifndef _XOPEN_SOURCE
#  define _XOPEN_SOURCE 600
#endif

#define JEM_INDEX_IS_INT

#define JEM_IDX_T     int
#define JEM_UIDX_T    unsigned int

#define JEM_IDX_C(i)  jem::idx_t  ( i )
#define JEM_UIDX_C(i) jem::uidx_t ( JEM_CONCAT(i,U) )

#endif
