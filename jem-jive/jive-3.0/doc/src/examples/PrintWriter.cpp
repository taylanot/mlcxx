
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

void printReport ( Ref<Writer>  output,
                   double       price,
                   int          count )
{
  Ref<PrintWriter>  printer = newInstance<PrintWriter> ( output );
  double            total   = price * count;

  printer->nformat.setIntegerWidth   ( 6 );
  printer->nformat.setFloatWidth     ( 6 );
  printer->nformat.setFractionDigits ( 4 );
  printer->setPageWidth              ( 60 );
  printer->setIndentWidth            ( 2 );

  print ( *printer,
          "Thank you for visiting ShopUntilYouDrop! ",
          "We hope that you will enjoy your new goodies ",
          "and that we will see you again soon.", endl );
  print ( *printer, blankLine,
          "Here is a summary of your purchase:",
          endl, indent, blankLine );
  print ( *printer,
          "Number of items ", dots( 24 ), count, endl );
  print ( *printer,
          "Price per item  ", dots( 24 ), price, endl );
  print ( *printer,
          "Total amount    ", dots( 24 ), total, endl );
  print ( *printer, blankLine, outdent,
          "Bye bye!", endl );
}
