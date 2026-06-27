import time
from Essentials.app import spellchecker

paragraph = '''“Nahseb tajtek ezempji biz zejjed ghaliex jien la se niehu dak li qed tghid int.”
“Jien nistaqsi, ghaliex din it-talba ma gietx accettata u ma regghetx marret lura ghand il-gvern?”
“Jien lilek lanqas biss nistmak ghaliex kull meta tippostja tghamel biex minghalik tinsolenta.”
“Imma kelli inhallas il-biljett jien ghaliex il KM ma tahdimx minn hemm.”
“Mela jien lanqas jista jkolli kunpens tal hsara li ghamluli dawn in nies.”
“Insomma nahseb xi kultant l-iskola saret wisq kollox jghaddi u hadd ma jimpurtah.”
“Ma nafx x naqbad nghid imma dan sa ftit taz zmien ilu ma kienx hekk.”
“Jien niftakar il-Mama tijaj Alla jahfrila tghamlilna dawn meta kienet tghamel xi torta.”
“X'tahsbu fuq din ma nahsibx jien li ghada tghamel sens li qassisin ma jistawx ikollom sess.”
“Daw it tip ta dghajes jinstabu ma dinja kolha ghalura malta ghax le.”'''

print("Starting test...")
t0 = time.time()
spellchecker.correct_text_rich(paragraph)
print(f'Time taken: {time.time() - t0:.3f}s')
