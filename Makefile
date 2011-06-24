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

LIBFLAGS = -I ./lib -I ./lib/pngwriter/include
LFLAGS = -lpng -lz -DNO_FREETYPE -lpngwriter
DEBUG = -ggdb
OPTIMIZE = -O3
ERROR = -Wconversion -Werror
CFLAGS = $(OPTIMIZE) -Wall -c $(DEBUG) $(ERROR) $(LIBFLAGS) $(LFLAGS)
LDFLAGS = $(OPTIMIZE) $(DEBUG) $(ERROR) -L ./lib/pngwriter/lib $(LFLAGS)

TARGET = terlR
INPUTEXT=pov
INPUTDIR=input
#INPUTFILE=bunny_small
INPUTFILE=simple_tri
OUTPUTDIR=images
OUTPUTEXT=png
WIDTH=640
HEIGHT=480
ARGS = -w $(WIDTH) -h $(HEIGHT) -i $(INPUTDIR)/$(INPUTFILE).$(INPUTEXT)

# Additional linker libraries
LIBS = $(LIBFLAGS)

# -------------------------------------------------------------
# Nothing should need changing below this line

# The source files
SRCS = $(wildcard src/*.cpp)
#SRCS = $(wildcard *.cpp)

OBJS = $(SRCS:.cpp=.o)

# Rules for building
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

.PHONY: lib
lib:
	$(shell) ./lib.sh

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
