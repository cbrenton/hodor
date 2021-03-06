# Original SDL-GL-basic Makefile template written by Hans de Ruiter
# Modified by Chris Brenton
#
# License:
# This source code can be used and/or modified without restrictions.
# It is provided as is and the author disclaims all warranties, expressed 
# or implied, including, without limitation, the warranties of
# merchantability and of fitness for any purpose. The user must assume the
# entire risk of using the Software.

CC     = g++
GPU_CC = nvcc
CP     = cp
RM     = rm -rf
KILL   = killall -9
SHELL  = /bin/sh
MAKE   = make

IFLAGS = -I./src -I./lib/objTester -I./lib -fopenmp -I/usr/local/include/ImageMagick-6
LFLAGS = -lpng -lz -lsfml-window -lsfml-graphics -lGL -lGLU -L./lib/objTester -lMagick++ -lMagickCore
DEBUG = -ggdb
OPTIMIZE = 
#ERROR = -Werror
ERROR = 
#CFLAGS = $(OPTIMIZE) -Wall -c $(DEBUG) $(ERROR) $(IFLAGS)
CFLAGS = $(OPTIMIZE) -c $(DEBUG) $(ERROR) $(IFLAGS) -Wno-deprecated
LDFLAGS = $(OPTIMIZE) $(DEBUG) $(ERROR) $(LFLAGS)

TARGET = hodor
INPUTEXT=obj
INPUTDIR=input
#INPUTFILE=bunny_small
INPUTFILE=test
OUTPUTDIR=images
OUTPUTEXT=tga
WIDTH=640
HEIGHT=640
ARGS = -g -w $(WIDTH) -h $(HEIGHT) -i $(INPUTDIR)/$(INPUTFILE).$(INPUTEXT) -p

# Additional linker libraries
LIBS = $(LIBFLAGS)

# -------------------------------------------------------------
# Nothing should need changing below this line

# The source files
#SRCS = $(wildcard src/*.cpp src/*/*.cpp)
#SRCS = $(wildcard src/*.cpp src/*/*.cpp src/*.h src/*/*.h)
SRCS = $(wildcard src/*.cpp src/structs/*.cpp src/geom/*.cpp src/img/*.cpp src/parse/*.cpp lib/objTester/*.cpp)

OBJS = $(SRCS:.cpp=.o)

# Rules for building
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

run:
	./$(TARGET) $(ARGS)

eog:
	eog ./$(OUTPUTDIR)/$(INPUTFILE).$(OUTPUTEXT)

test:	run eog

pov:
	vim ./$(INPUTDIR)/$(INPUTFILE).$(INPUTEXT)

gdb:
	gdb --args ./$(TARGET) $(ARGS)

vg:	valgrind

valgrind:
	valgrind --tool=memcheck --leak-check=full ./$(TARGET) $(ARGS)

clean:
	$(RM) $(TARGET) $(OBJS)

killall:
	$(KILL) $(TARGET)
