#
# Compiler flags
#
CC     = nvcc#g++
CFLAGS = --compiler-options -Wall --compiler-options -Werror --compiler-options -Wextra --compiler-options -Wno-unused
LIBS	= -lpthread -lsfml-system -lsfml-window -lsfml-graphics -lGLEW -lGL -lGLU -lglut -lgmp -lgmpxx
#
# Project files
#
SRCS_CPP = $(wildcard *.cpp)
SRCS_CU = $(wildcard *.cu)
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS_CU = $(SRCS_CU:.cu=.cu.o)

OBJS = $(OBJS_CPP) $(OBJS_CU)

EXE  = MEXP

#
# Debug build settings
#
DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXE)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJS))
DBGCFLAGS = -g -O0 -DDEBUG

#
# Release build settings
#
RELDIR = release
RELEXE = $(RELDIR)/$(EXE)
RELOBJS = $(addprefix $(RELDIR)/, $(OBJS))
RELCFLAGS = -O3 -DNDEBUG

.PHONY: all clean debug prep release remake run rund

# Default build
all: prep release

#
# Debug rules
#
debug: $(DBGEXE)

$(DBGEXE): $(DBGOBJS)
	$(CC) $(CFLAGS) $(DBGCFLAGS) -o $(DBGEXE) $^ $(LIBS)

$(DBGDIR)/%.o: %.cpp
	$(CC) -c $(CFLAGS) $(DBGCFLAGS) -o $@ $<

$(DBGDIR)/%.cu.o: %.cu
	$(CC) -c $(CFLAGS) $(DBGCFLAGS) -o $@ $<


#
# Release rules
#
release: $(RELEXE)

$(RELEXE): $(RELOBJS)
	$(CC) $(CFLAGS) $(RELCFLAGS) -o $(RELEXE) $^ $(LIBS)

$(RELDIR)/%.o: %.cpp
	$(CC) -c $(CFLAGS) $(RELCFLAGS) -o $@ $<

$(RELDIR)/%.cu.o: %.cu
	$(CC) -c $(CFLAGS) $(RELCFLAGS) -o $@ $<


#
# Other rules
#
prep:
	@mkdir -p $(DBGDIR) $(RELDIR)

remake: clean all

clean:
	rm -f $(RELEXE) $(RELOBJS) $(DBGEXE) $(DBGOBJS) $(RELDIR)/*.o $(DBGDIR)/*.o
run:
	$(RELEXE)

rund:
	$(DBGEXE)
