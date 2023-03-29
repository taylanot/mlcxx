
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/native/Process.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/Process.h>


JEM_DEFINE_CLASS( jem::Process );


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class Process
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Process::KILLED = -1;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Process::~Process ()
{}


//-----------------------------------------------------------------------
//   exec
//-----------------------------------------------------------------------


void Process::exec

  ( const Array<String>&  cline,
    Options               options )

{
  JEM_PRECHECK2 ( cline.size(), "empty command line" );

  NativeProcess::exec ( cline, options );
}


//-----------------------------------------------------------------------
//   spawn
//-----------------------------------------------------------------------


Ref<Process> Process::spawn

  ( const Array<String>&  cline,
    Options               options )

{
  JEM_PRECHECK2 ( cline.size(), "empty command line" );

  SpawnParams  params;

  return newInstance<NativeProcess> ( cline, params, options );
}


Ref<Process> Process::spawn

  ( const Array<String>&  cline,
    const SpawnParams&    params,
    Options               options )

{
  JEM_PRECHECK2 ( cline.size(), "empty command line" );

  if ( params.stdInput == SpawnParams::STD_OUT )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "invalid spawn parameters; "
      "stdin can not be redirected to stdout"
    );
  }

  if ( params.stdOutput == SpawnParams::STD_OUT )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "invalid spawn parameters; "
      "stdout can not be redirected to itself"
    );
  }

  return newInstance<NativeProcess> ( cline, params, options );
}


//-----------------------------------------------------------------------
//   getExecPath
//-----------------------------------------------------------------------


String Process::getExecPath ()
{
  return NativeProcess::getExecPath ();
}


//=======================================================================
//   class SpawnParams
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SpawnParams::INHERIT  = "";
const char*  SpawnParams::DEV_NULL = JEM_DEV_NULL;
const char*  SpawnParams::PIPE     = "|";
const char*  SpawnParams::STD_OUT  = ">1";


JEM_END_PACKAGE_BASE
