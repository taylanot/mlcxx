
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


#include <jive/model/Actions.h>


JIVE_BEGIN_PACKAGE( model )


//=======================================================================
//   class Actions
//=======================================================================


#undef  JIVE_DEFINE_ACTION
#define JIVE_DEFINE_ACTION( ACTION )       \
                                           \
const char*  Actions::ACTION = #ACTION


JIVE_DEFINE_ACTION( INIT              );
JIVE_DEFINE_ACTION( SHUTDOWN          );
JIVE_DEFINE_ACTION( NEW_MATRIX0       );
JIVE_DEFINE_ACTION( NEW_MATRIX1       );
JIVE_DEFINE_ACTION( NEW_MATRIX2       );
JIVE_DEFINE_ACTION( UPD_MATRIX0       );
JIVE_DEFINE_ACTION( UPD_MATRIX1       );
JIVE_DEFINE_ACTION( UPD_MATRIX2       );
JIVE_DEFINE_ACTION( GET_MATRIX0       );
JIVE_DEFINE_ACTION( GET_MATRIX1       );
JIVE_DEFINE_ACTION( GET_MATRIX2       );
JIVE_DEFINE_ACTION( NEW_PRECON0       );
JIVE_DEFINE_ACTION( NEW_PRECON1       );
JIVE_DEFINE_ACTION( NEW_PRECON2       );
JIVE_DEFINE_ACTION( UPD_PRECON0       );
JIVE_DEFINE_ACTION( UPD_PRECON1       );
JIVE_DEFINE_ACTION( UPD_PRECON2       );
JIVE_DEFINE_ACTION( GET_PRECON0       );
JIVE_DEFINE_ACTION( GET_PRECON1       );
JIVE_DEFINE_ACTION( GET_PRECON2       );
JIVE_DEFINE_ACTION( GET_INT_VECTOR    );
JIVE_DEFINE_ACTION( GET_EXT_VECTOR    );
JIVE_DEFINE_ACTION( GET_CONSTRAINTS   );
JIVE_DEFINE_ACTION( GET_MP_MODEL      );
JIVE_DEFINE_ACTION( GET_SOLVER_PARAMS );
JIVE_DEFINE_ACTION( GET_RES_SCALE     );
JIVE_DEFINE_ACTION( GET_BOUNDS        );
JIVE_DEFINE_ACTION( UPDATE            );
JIVE_DEFINE_ACTION( COMMIT            );
JIVE_DEFINE_ACTION( CHECK_CONVERGED   );
JIVE_DEFINE_ACTION( CHECK_COMMIT      );
JIVE_DEFINE_ACTION( CANCEL            );
JIVE_DEFINE_ACTION( ADVANCE           );
JIVE_DEFINE_ACTION( GET_TABLE         );
JIVE_DEFINE_ACTION( SET_LOAD_SCALE    );

#undef JIVE_DEFINE_ACTION


const char*  Actions::NEW_MATRIX[3] = { NEW_MATRIX0,
                                        NEW_MATRIX1,
                                        NEW_MATRIX2 };

const char*  Actions::UPD_MATRIX[3] = { UPD_MATRIX0,
                                        UPD_MATRIX1,
                                        UPD_MATRIX2 };

const char*  Actions::GET_MATRIX[3] = { GET_MATRIX0,
                                        GET_MATRIX1,
                                        GET_MATRIX2 };

const char*  Actions::NEW_PRECON[3] = { NEW_PRECON0,
                                        NEW_PRECON1,
                                        NEW_PRECON2 };

const char*  Actions::UPD_PRECON[3] = { UPD_PRECON0,
                                        UPD_PRECON1,
                                        UPD_PRECON2 };

const char*  Actions::GET_PRECON[3] = { GET_PRECON0,
                                        GET_PRECON1,
                                        GET_PRECON2 };


//=======================================================================
//   class ActionParams
//=======================================================================


#undef  JIVE_DEFINE_ACTIONPARAM
#define JIVE_DEFINE_ACTIONPARAM( PARAM, param )  \
                                                 \
const char*  ActionParams::PARAM = #param


JIVE_DEFINE_ACTIONPARAM( MATRIX0,        matrix0       );
JIVE_DEFINE_ACTIONPARAM( MATRIX1,        matrix1       );
JIVE_DEFINE_ACTIONPARAM( MATRIX2,        matrix2       );
JIVE_DEFINE_ACTIONPARAM( PRECON0,        precon0       );
JIVE_DEFINE_ACTIONPARAM( PRECON1,        precon1       );
JIVE_DEFINE_ACTIONPARAM( PRECON2,        precon2       );
JIVE_DEFINE_ACTIONPARAM( INT_VECTOR,     intVector     );
JIVE_DEFINE_ACTIONPARAM( EXT_VECTOR,     extVector     );
JIVE_DEFINE_ACTIONPARAM( CONSTRAINTS,    constraints   );
JIVE_DEFINE_ACTIONPARAM( MP_MODEL,       mpModel       );
JIVE_DEFINE_ACTIONPARAM( TABLE,          table         );
JIVE_DEFINE_ACTIONPARAM( TABLE_NAME,     tableName     );
JIVE_DEFINE_ACTIONPARAM( TABLE_ROWS,     tableRows     );
JIVE_DEFINE_ACTIONPARAM( TABLE_WEIGHTS,  tableWeights  );
JIVE_DEFINE_ACTIONPARAM( LOAD_CASE,      loadCase      );
JIVE_DEFINE_ACTIONPARAM( SCALE_FACTOR,   scaleFactor   );
JIVE_DEFINE_ACTIONPARAM( ACCEPT,         accept        );
JIVE_DEFINE_ACTIONPARAM( CONVERGED,      converged     );
JIVE_DEFINE_ACTIONPARAM( RESIDUAL,       residual      );
JIVE_DEFINE_ACTIONPARAM( RES_SCALE,      rscale        );
JIVE_DEFINE_ACTIONPARAM( LOWER_BOUND,    lowerBound    );
JIVE_DEFINE_ACTIONPARAM( UPPER_BOUND,    upperBound    );

#undef JIVE_DEFINE_ACTIONPARAM


const char*  ActionParams::MATRIX[3] = { MATRIX0,
                                         MATRIX1,
                                         MATRIX2 };

const char*  ActionParams::PRECON[3] = { PRECON0,
                                         PRECON1,
                                         PRECON2 };


JIVE_END_PACKAGE( model )
