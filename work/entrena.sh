#! /bin/bash

if [[ $# != 5 ]]; then
	echo "Empleo: $0 momentum tam_batch lr"
	exit
fi

NUM_NOD=2048
NUM_NOD=512

NUM_CAP=5
DIM_INT=256

NOME=adam.uno

DIR_WRK=.

set -o pipefail

DIR_LOG=$DIR_WRK/Log
FIC_LOG=$DIR_LOG/$(basename $0 .sh).$NOME.log
FIC_LOG=$DIR_LOG/$(basename $0 .sh).$NOME.log
[ -d $DIR_LOG ] || mkdir -p $DIR_LOG

exec > >(/home/grauvi/TFG/Joan-main/src/teeqdm $FIC_LOG) 2>&1

#for LR in 1e-5 1e-4 ; do
for LR in $3 ; do

for MOMENTUM in $1 ; do

for TAM_BATCH in $2 ; do

for MODELO in $4 ; do

for OPTIM in $5 ; do

if [[ $TAM_BATCH == 100 ]]; then
	NUM_EPO=10
else
	NUM_EPO=10
fi

MODELO=$4

OPTIM=$5


NOM_EXP=$NOME.$TAM_BATCH

# Seleccion de la red neuronal

if [ $MODELO == MLP ]; then
	NUM_CAP=${NUM_CAP:-5}
	DIM_INT=${DIM_INT:-256}

	NOM_EXP=$NOM_EXP.$MODELO.$NUM_CAP.$DIM_INT

	red="mlp_N(numCap=$NUM_CAP, dimIni=numCof, dimSal=tamVoc, dimInt=$DIM_INT)"
elif [ $MODELO == DS ]; then
	NUM_NOD=${NUM_NOD:-2048}

	NOM_EXP=$NOM_EXP.$MODELO.$NUM_NOD

	red="DeepSpeech(n_feature=numCof, n_hidden=$NUM_NOD, n_class=tamVoc)"
fi

# Seleccion del optimizador

if [ $OPTIM == Adam ]; then
	BETA1=${BETA1:-$MOMENTUM}
	BETA2=${BETA2:-999}

	if [ $BETA1 == 99 ]; then
		BETA2=999
	fi

	NOM_EXP=$NOM_EXP.$OPTIM.$LR.$BETA1.$BETA2

	EXEC_PRE="conf/mod_ls.py,conf/topos.py"

	Optim="lambda params: torch.optim.Adam(params, lr=$LR, betas=(0.$BETA1, 0.$BETA2))"
elif [ $OPTIM == SGD ]; then
	MOMENTUM=${MOMENTUM:-99}

	NOM_EXP=$NOM_EXP.$OPTIM.$LR.$MOMENTUM

	EXEC_PRE="conf/mod_ls.py,conf/topos.py"

	Optim="lambda params: torch.optim.SGD(params, lr=$LR, momentum=0.$MOMENTUM, dampening=0.$MOMENTUM)"
elif [ $OPTIM == Acme ]; then
	BETA1=${MOMENTUM:-99}
	BETA2=999
	if [ $BETA1 == 99 ]; then
		BETA2=999
	fi

	EXEC_PRE="conf/mud_ls.py,conf/topos.py,../src/$NOME.py"

	NOM_EXP=$NOM_EXP.$OPTIM.$LR.$BETA1

	Optim="lambda params: Acme(params, lr=$LR, beta=0.$BETA1, foreach=True)"
fi

# Ficheros guía

DIR_GUI=$DIR_WRK/Gui

GUI_ENT=$DIR_GUI/train-red.gui
GUI_DEV=$DIR_GUI/train-red.gui

GUI_ENT=$DIR_GUI/train-clean-100.gui
GUI_DEV=$DIR_GUI/dev-clean.gui

# Directorios de las señales

DIR_PRM=$DIR_WRK/Prm
DIR_TRN=$DIR_WRK/Trn

# Lista de unidades a reconocer

LIS_UNI=$DIR_WRK/Lis/fonemas.lis

FIC_LOG=$DIR_LOG/$(basename $0 .sh).$NOM_EXP.log

hostname |/home/grauvi/TFG/Joan-main/src/teeqdm $FIC_LOG
pwd | /home/grauvi/TFG/Joan-main/src/teeqdm -a $FIC_LOG
date | /home/grauvi/TFG/Joan-main/src/teeqdm -a $FIC_LOG

FIC_EVO=$DIR_WRK/Evo/$NOM_EXP.evo
FIC_MOD=$DIR_WRK/Mod/$NOM_EXP.mod

lotesEnt="-E 'lotesLS(\"$DIR_PRM\", \"$DIR_TRN\", \"$LIS_UNI\", $TAM_BATCH, \"$GUI_ENT\")'"
lotesDev="-D 'lotesLS(\"$DIR_PRM\", \"$DIR_TRN\", \"$LIS_UNI\", 1, \"$GUI_DEV\")'"
execPre="-x $EXEC_PRE"
ficLog=$FIC_EVO
modelo="-M 'ModLS(ficLisUni=\"$LIS_UNI\", red=$red, Optim=$Optim, ficLog=\"$ficLog\", device=\"cuda\")'"
numEpo="-e $NUM_EPO -v 10"

EXEC="../src/entrena.py $execPre $lotesEnt $lotesDev $modelo $numEpo"
echo $EXEC | /home/grauvi/TFG/Joan-main/src/teeqdm -a $FIC_LOG; bash -c "$EXEC" 2>&1 | /home/grauvi/TFG/Joan-main/src/teeqdm -a $FIC_LOG || exit 1

date
echo s\'acabao

done
done
done
done
done


date
echo s\'acabao del todo
