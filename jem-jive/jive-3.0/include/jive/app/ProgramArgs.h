
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

#ifndef JIVE_APP_PROGRAMARGS_H
#define JIVE_APP_PROGRAMARGS_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class ProgramArgs
//-----------------------------------------------------------------------


class ProgramArgs : public Object,
                    public jem::Clonable
{
 public:

  JEM_DECLARE_CLASS     ( ProgramArgs, Object );


  explicit                ProgramArgs

    ( const String&         arg0 = "" );

                          ProgramArgs

    ( const String&         arg0,
      const String&         arg1 );

                          ProgramArgs

    ( const String&         arg0,
      const String&         arg1,
      const String&         arg2 );

  explicit                ProgramArgs

    ( const StringVector&   args );

                          ProgramArgs

    ( int                   argc,
      char**                argv );

                          ProgramArgs

    ( const Self&           rhs );

  virtual Ref<Object>     clone           () const override;
  Ref<Self>               dup             () const;

  int&                    argc            ();
  inline int              argc            () const;
  char**                  argv            ();
  inline char* const*     argv            () const;
  void                    commit          ();
  StringVector            toArray         () const;
  String                  getProgramName  () const;

  void                    store

    ( const Properties&     globdat )        const;

  static Ref<Self>        get

    ( const Properties&     globdat );

  static Ref<Self>        find

    ( const Properties&     globdat );


 protected:

  virtual                ~ProgramArgs     ();


 private:

  void                    alloc_

    ( int                   argc );

  void                    free_           ();

  void                    append_

    ( const String&         arg );


 private:

  int                     newArgc_;
  char**                  newArgv_;

  int                     oldArgc_;
  char**                  oldArgv_;

  int                     orgArgc_;
  char**                  orgArgv_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   argc
//-----------------------------------------------------------------------


inline int ProgramArgs::argc () const
{
  return oldArgc_;
}


//-----------------------------------------------------------------------
//   argv
//-----------------------------------------------------------------------


inline char* const* ProgramArgs::argv () const
{
  return oldArgv_;
}


JIVE_END_PACKAGE( app )

#endif
