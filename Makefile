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

IFLAGS = -I./src -I./lib -I./lib/pngwriter/include -DNO_FREETYPE
LFLAGS = -lpng -lz -lpngwriter -L./lib/pngwriter/lib
DEBUG = -ggdb
OPTIMIZE = -O3
ERROR = -Wconversion -Werror
CFLAGS = $(OPTIMIZE) -Wall -c $(DEBUG) $(ERROR) $(IFLAGS)
LDFLAGS = $(OPTIMIZE) $(DEBUG) $(ERROR) $(LFLAGS)

TARGET = terlR
INPUTEXT=pov
INPUTDIR=input
#INPUTFILE=bunny_small
INPUTFILE=cornell_box
OUTPUTDIR=images
OUTPUTEXT=png
WIDTH=640
HEIGHT=640
ARGS = -a 4 -g -w $(WIDTH) -h $(HEIGHT) -i $(INPUTDIR)/$(INPUTFILE).$(INPUTEXT)

# Additional linker libraries
LIBS = $(LIBFLAGS)

# -------------------------------------------------------------
# Nothing should need changing below this line

# The source files
SRCS = $(wildcard src/*.cpp src/*/*.cpp)
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
