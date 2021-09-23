ICPP = g++
IC = gcc
INCLUDE = -I.
OPT = -O2

OBJS1 = random.o langevin.o Ito_integrator.o
OBJS2 = random.o committor.o Ito_integrator.o
LIBS_COMMON = -lgsl -lgslcblas -I/opt/local/include -L/opt/local/lib

main: 	$(OBJS1)
	$(ICPP) $(OPT) -o ../langevin $(OBJS1) $(LIBS_COMMON)

committor: 	$(OBJS2)
	$(ICPP) $(OPT) -o ../committor $(OBJS2) $(LIBS_COMMON)

%.o: %.cpp
	$(ICPP) $(OPT) $(DIRECT) -c $(INCLUDE) $(LIBS_COMMON) $(<) $(outputs)

clean:
	rm -fv *.o langevin
