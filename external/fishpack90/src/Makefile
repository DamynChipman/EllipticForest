
include ../make.inc

SRC=fish.f blktri.f cblktri.f cmgnbn.f comf.f fftpack.f \
    genbun.f gnbnaux.f hstcrt.f hstcsp.f hstcyl.f hstplr.f \
    hstssp.f hw3crt.f hwscrt.f hwscsp.f hwscyl.f hwsplr.f \
    hwsssp.f pois3d.f poistg.f sepaux.f sepeli.f sepx4.f

OBJ=$(subst .f,.o,$(SRC))
OBJS=$(addprefix ../objs/,$(OBJ))


$(LIB) : $(OBJS)
	$(AR) -rv $@ $? 

fish.f : fish.F
#$(CPP) $< > $@
	$(CPP)

../objs/fish.o : fish.f
	$(F90) -c $< -o $@

../objs/%.o : %.f
	$(F90) -c $< -o $@

clean:
	rm -f $(LIB) $(OBJS) ../lib/fish.mod ../lib/FISH.mod fish.o fish.s
