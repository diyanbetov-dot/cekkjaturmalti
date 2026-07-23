# -*- coding: utf-8 -*-
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
os.environ['SPELLCHECK_CORPUS_SCORING'] = 'true'
os.environ['SPELLCHECK_CORPUS_UNIGRAM'] = 'true'
os.environ['SPELLCHECK_CORPUS_BIGRAM'] = 'true'
os.environ['SPELLCHECK_CORPUS_MAX_SCORE'] = '0.25'
sys.path.insert(0, os.path.abspath('.'))
from Essentials.app import spellchecker
text = '''Gheziez hbieb,

Naf li bhalissa ghaddejin minn zmien difficcli u iebes minhabba nuqqas ta' dawl ghall hinijiet twal u eccessivi. Shana, twahhil, tidlik u hassazinijat. Naf ukoll li hawn hafna minn qed ibaghti minhabba eta, dizabilta' jew kundizzjonijiet medici. U ta' dan jiddispjacini hafna, ghaliex hadd ma haqqu jghaddi minn dan kollu!



Jin kulma nixtieq huwa haga wahda biss: forsi nkun ftit biased ukoll izda nhoss li ghandi nghidha. Jekk jghogbokom tihduwiex mal-haddiema tal-Enemalta jew il-customer care taghhom. Naf u nifhem li mhux sitwazzjoni sabiha u pjacevoli, izda dawn mghandhom l-ebda tort! Il-haddiema kollha hilom jahdmu fix-xemx u s-shana granet twal hafna u bla waqfien. U ghalhekk nixtieq li tipruvaw tifmu li huma mghandhomx tort u qed jipruvaw bil-kapacita' kollha li ghandhom jreggghu kollox lura ghan-normal. It-tort tuh lin-nies ta' fuq; dawk li jikkmandaw il-ligi! 



Dawn il-haddiema li qed jahdmu lejl u nhar bix ituna l-lura l-aktar haga bazika li tezisti: il-kumdita', ma jahtu xejn!


 Miniex nistenna li ma ccemplux jew ma tistaqsux ghax ghandkom dritt- izda li titfu htija fuq min mghandux huwa ingust u jwegga' hafna. Missieri flimkien mal-haddiema kollha tal-enemalta hilhom granet u ijlieli shah jahdmu bla nifs u bilkemm jistriehu jew narawhom. Ejjew ma nhalllux is-sahna u r-rabja tal-mument tirkibna billi nghajjru jew imqaddru lil haddiema li jahdmu mill-qalb f'dan it-temp kifer!


Min hawn nghid grazzi lil haddiema kollha tal-enemalta u l-customer care li qed jaghmlu hilithom kollha bix isolvu problema li mghandhomx tort fiha!'''
rich_result = spellchecker.correct_text_rich(text)
output = {
    'corrected_text': rich_result.get('corrected_text', ''),
    'tokens': rich_result.get('tokens', []),
    'corpus_status': getattr(getattr(spellchecker, 'corpus_scorer', None), 'get_status_details', lambda: None)(),
    'has_bertu': hasattr(spellchecker, 'bertu_reranker'),
    'bertu_available': getattr(getattr(spellchecker, 'bertu_reranker', None), '_available', None),
}
with open('scratch_compare_experimental.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print('EXPERIMENTAL_DONE')
