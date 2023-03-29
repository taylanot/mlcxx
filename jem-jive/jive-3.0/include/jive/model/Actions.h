
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

#ifndef JIVE_MODEL_ACTIONS_H
#define JIVE_MODEL_ACTIONS_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( model )


//-----------------------------------------------------------------------
//   class Actions
//-----------------------------------------------------------------------


class Actions
{
 public:

  static const char*        INIT;
  static const char*        SHUTDOWN;

  static const char*        NEW_MATRIX[3];
  static const char*        NEW_MATRIX0;
  static const char*        NEW_MATRIX1;
  static const char*        NEW_MATRIX2;

  static const char*        UPD_MATRIX[3];
  static const char*        UPD_MATRIX0;
  static const char*        UPD_MATRIX1;
  static const char*        UPD_MATRIX2;

  static const char*        GET_MATRIX[3];
  static const char*        GET_MATRIX0;
  static const char*        GET_MATRIX1;
  static const char*        GET_MATRIX2;

  static const char*        NEW_PRECON[3];
  static const char*        NEW_PRECON0;
  static const char*        NEW_PRECON1;
  static const char*        NEW_PRECON2;

  static const char*        UPD_PRECON[3];
  static const char*        UPD_PRECON0;
  static const char*        UPD_PRECON1;
  static const char*        UPD_PRECON2;

  static const char*        GET_PRECON[3];
  static const char*        GET_PRECON0;
  static const char*        GET_PRECON1;
  static const char*        GET_PRECON2;

  static const char*        GET_INT_VECTOR;
  static const char*        GET_EXT_VECTOR;
  static const char*        GET_CONSTRAINTS;
  static const char*        GET_MP_MODEL;
  static const char*        GET_SOLVER_PARAMS;
  static const char*        GET_RES_SCALE;
  static const char*        GET_BOUNDS;

  static const char*        UPDATE;
  static const char*        CHECK_CONVERGED;
  static const char*        CHECK_COMMIT;
  static const char*        COMMIT;
  static const char*        CANCEL;
  static const char*        ADVANCE;

  static const char*        GET_TABLE;
  static const char*        SET_LOAD_SCALE;

};


//-----------------------------------------------------------------------
//   class ActionParams
//-----------------------------------------------------------------------


class ActionParams
{
 public:

  static const char*        MATRIX[3];
  static const char*        MATRIX0;
  static const char*        MATRIX1;
  static const char*        MATRIX2;

  static const char*        PRECON[3];
  static const char*        PRECON0;
  static const char*        PRECON1;
  static const char*        PRECON2;

  static const char*        INT_VECTOR;
  static const char*        EXT_VECTOR;
  static const char*        CONSTRAINTS;
  static const char*        MP_MODEL;

  static const char*        TABLE;
  static const char*        TABLE_NAME;
  static const char*        TABLE_ROWS;
  static const char*        TABLE_WEIGHTS;

  static const char*        LOAD_CASE;
  static const char*        SCALE_FACTOR;

  static const char*        ACCEPT;
  static const char*        CONVERGED;
  static const char*        RESIDUAL;
  static const char*        RES_SCALE;

  static const char*        LOWER_BOUND;
  static const char*        UPPER_BOUND;

};


JIVE_END_PACKAGE( model )

#endif
