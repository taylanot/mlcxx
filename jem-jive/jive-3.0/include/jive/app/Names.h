
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_APP_NAMES_H
#define JIVE_APP_NAMES_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class RunvarNames
//-----------------------------------------------------------------------


class RunvarNames
{
 public:

  static const char*    LOAD_SCALE;
  static const char*    RUNTIME;

};


//-----------------------------------------------------------------------
//   class PropertyNames
//-----------------------------------------------------------------------


class PropertyNames
{
 public:

  static const char*    APPEND;
  static const char*    BACKUPS;
  static const char*    BUFSIZE;
  static const char*    CMD_FILE;
  static const char*    DATA_SETS;
  static const char*    DUMP_COND;
  static const char*    EXPR;
  static const char*    FILE;
  static const char*    FG_MODE;
  static const char*    HEADER;
  static const char*    INPUT_MODULE;
  static const char*    ITER;
  static const char*    LOAD_CASE;
  static const char*    LOG;
  static const char*    MODEL;
  static const char*    MODULE;
  static const char*    MODULES;
  static const char*    NAME;
  static const char*    NEW_MODULES;
  static const char*    NOISE_LEVEL;
  static const char*    PAGE_WIDTH;
  static const char*    PARAMS;
  static const char*    PATTERN;
  static const char*    PAUSE;
  static const char*    PRECISION;
  static const char*    PROMPT;
  static const char*    RANK;
  static const char*    RUN_COND;
  static const char*    SAMPLE_COND;
  static const char*    SAVE_COND;
  static const char*    SCALE_FUNC;
  static const char*    SEPARATOR;
  static const char*    SYNC;
  static const char*    TABLES;
  static const char*    TEMP;
  static const char*    VECTORS;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef RunvarNames     VarNames;
typedef PropertyNames   PropNames;


JIVE_END_PACKAGE( app )

#endif
