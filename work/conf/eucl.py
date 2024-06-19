from euclideo import ModEucl, lotesEucl

lotesEnt = lotesEucl('Prm', 'Sen', 'Gui/train.gui')
lotesDev = lotesEucl('Prm', 'Sen', 'Gui/train.gui')
lotesRec = lotesEucl('Prm', None, 'Gui/train.gui')

ficMod = 'modelo.mod'

modEnt = ModEucl(ficLisUni='Lis/vocales.lis')
modRec = ModEucl(ficMod=ficMod)
