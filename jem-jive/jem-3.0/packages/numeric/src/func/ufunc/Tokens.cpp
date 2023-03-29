
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


#include "Tokens.h"


JEM_BEGIN_UFUNC_IMP


//=======================================================================
//   class Tokens
//=======================================================================

//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


#undef  JEM_CASE_TOKEN
#define JEM_CASE_TOKEN( TOK, str )              \
                                                \
  case TOK: tokstr = str; break


String Tokens::toString ( int token )
{
  if ( token >= 0 )
  {
    return String ( (char) token );
  }

  const char*  tokstr = 0;

  switch ( token )
  {
  JEM_CASE_TOKEN( NONE,    "<none>"   );
  JEM_CASE_TOKEN( END,     "<END>"    );
  JEM_CASE_TOKEN( SYMBOL,  "<symbol>" );
  JEM_CASE_TOKEN( NUMBER,  "<number>" );
  JEM_CASE_TOKEN( SAVE,    "save"     );
  JEM_CASE_TOKEN( LET,     "let"      );
  JEM_CASE_TOKEN( IF,      "if"       );
  JEM_CASE_TOKEN( RETURN,  "return"   );
  JEM_CASE_TOKEN( ELSE,    "else"     );
  JEM_CASE_TOKEN( ELIF,    "elif"     );
  JEM_CASE_TOKEN( ENDIF,   "endif"    );
  JEM_CASE_TOKEN( PI,      "PI"       );
  JEM_CASE_TOKEN( EPS,     "EPS"      );
  JEM_CASE_TOKEN( BIGNUM,  "HUGE"     );
  JEM_CASE_TOKEN( TINY,    "TINY"     );
  JEM_CASE_TOKEN( TRUE,    "true"     );
  JEM_CASE_TOKEN( FALSE,   "false"    );
  JEM_CASE_TOKEN( OR,      "||"       );
  JEM_CASE_TOKEN( AND,     "&&"       );
  JEM_CASE_TOKEN( LE,      "<="       );
  JEM_CASE_TOKEN( GE,      ">="       );
  JEM_CASE_TOKEN( ABS,     "abs"      );
  JEM_CASE_TOKEN( SQRT,    "sqrt"     );
  JEM_CASE_TOKEN( CEIL,    "ceil"     );
  JEM_CASE_TOKEN( FLOOR,   "floor"    );
  JEM_CASE_TOKEN( POW,     "pow"      );
  JEM_CASE_TOKEN( EXP,     "exp"      );
  JEM_CASE_TOKEN( LOG,     "log"      );
  JEM_CASE_TOKEN( LOG10,   "log10"    );
  JEM_CASE_TOKEN( SIN,     "sin"      );
  JEM_CASE_TOKEN( COS,     "cos"      );
  JEM_CASE_TOKEN( TAN,     "tan"      );
  JEM_CASE_TOKEN( ASIN,    "asin"     );
  JEM_CASE_TOKEN( ACOS,    "acos"     );
  JEM_CASE_TOKEN( MAX,     "max"      );
  JEM_CASE_TOKEN( MIN,     "min"      );
  JEM_CASE_TOKEN( DEFINED, "defined"  );

  default: tokstr = "<unknown>";
  }

  return String ( tokstr );
}


JEM_END_UFUNC_IMP
