
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


#include <GLFW/glfw3.h>

static void keyCallback

  ( GLFWwindow*  window,
    int          key,
    int          scancode,
    int          action,
    int          mods )

{
  if ( (key == GLFW_KEY_ESCAPE) && (action == GLFW_PRESS) )
  {
    glfwSetWindowShouldClose ( window, GLFW_TRUE );
  }
}

int main ( int argc, char** argv )
{
  GLFWwindow*  window;

  if ( ! glfwInit() )
  {
    return 1;
  }

  window = glfwCreateWindow ( 640, 480, "Test", nullptr, nullptr );

  if ( ! window )
  {
    glfwTerminate ();

    return 1;
  }

  glfwSetKeyCallback     ( window, keyCallback );
  glfwMakeContextCurrent ( window );
  glfwSwapInterval       ( 1 );

  while ( ! glfwWindowShouldClose( window ) )
  {
    glfwSwapBuffers ( window );
    glfwPollEvents  ();
  }

  glfwDestroyWindow ( window );
  glfwTerminate     ();

  return 0;
}

