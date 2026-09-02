#!/usr/bin/env python3
"""Plot confirmed noise metrics and write an English results note, entirely offline.

Requires matplotlib. Run from the project root; point --metrics-dir at the exact
completed directory produced by calculate_noise_metrics.py. No newest-run guessing.
Inputs and existing figures are never overwritten. A figures_manifest.json is
written last; its absence indicates an incomplete export.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import sys
import uuid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data):
    require(data.get('script_version') == 'noise_metrics_v1.0', 'Unsupported metrics version')
    rows = sorted((r for r in data['by_level'] if r['subset'] == 'all'), key=lambda r:r['noise_fraction'])
    clean = sorted((r for r in data['by_level'] if r['subset'] == 'clean'), key=lambda r:r['noise_fraction'])
    levels = [0.0, 0.2, 0.4, 0.6]
    require([r['noise_fraction'] for r in rows] == levels, 'Missing or duplicated all-sample conditions')
    require([r['noise_fraction'] for r in clean] == levels, 'Missing or duplicated Clean conditions')
    for subset, count in ((rows,30),(clean,29)):
        for r in subset:
            require(r['planned_count'] == count, 'Unexpected condition size')
            keys = ['correct_count','wrong_answered_count','abstain_count','schema_failure_count']
            require(all(isinstance(r[k],int) and r[k]>=0 for k in keys), 'Invalid outcome counts')
            require(sum(r[k] for k in keys)==count, 'Outcome counts do not sum to the planned total')
            require(r['answered_count'] == r['correct_count']+r['wrong_answered_count'], 'Answered count mismatch')
            require(r['valid_output_count'] == r['answered_count']+r['abstain_count'], 'Valid count mismatch')
            require(math.isclose(r['end_to_end_accuracy'],r['correct_count']/count), 'Accuracy denominator mismatch')
            require(0 <= r['citation_supported_count'] <= r['citation_applicable_count'] <= r['valid_output_count'], 'Invalid citation counts')
            support = r['citation_support_accuracy_applicable']
            if r['citation_applicable_count']:
                require(math.isclose(support,r['citation_supported_count']/r['citation_applicable_count']), 'Citation denominator mismatch')
            else:
                require(support is None, 'Undefined citation rate must be null')
    paired = sorted((r for r in data['paired_vs_zero'] if r['subset']=='all'), key=lambda r:r['noise_fraction'])
    require([r['noise_fraction'] for r in paired] == levels[1:], 'Missing paired comparisons')
    for p, r in zip(paired,rows[1:]):
        require(p['paired_count']==30 and sum(p[k] for k in ['both_correct','correct_to_incorrect','incorrect_to_correct','both_incorrect'])==30, 'Invalid paired totals')
        require(p['both_correct']+p['correct_to_incorrect']==rows[0]['correct_count'], 'Paired baseline mismatch')
        require(p['both_correct']+p['incorrect_to_correct']==r['correct_count'], 'Paired comparison mismatch')
    return rows,clean,paired


def load(directory):
    manifest_bytes = (directory/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get('complete') is True, 'Metrics export is incomplete')
    require(manifest.get('script_version') == 'noise_metrics_v1.0', 'Unexpected manifest version')
    inputs = {}
    for name, digest in manifest['output_sha256'].items():
        require(Path(name).name == name and '/' not in name and '\\' not in name, 'Invalid manifest path')
        value = (directory/name).read_bytes()
        require(sha(value)==digest, 'File changed after metrics export: '+name)
        inputs[name] = value
    require('noise_metrics.json' in inputs, 'Manifest does not include noise_metrics.json')
    data = json.loads(inputs['noise_metrics.json'])
    validate(data)
    hashes = {name:sha(value) for name,value in inputs.items()}
    hashes['manifest.json'] = sha(manifest_bytes)
    return data, manifest, hashes


def draw(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#263344','text.color':'#263344',
                         'xtick.color':'#465468','ytick.color':'#465468','svg.fonttype':'none'})
    fig,(left,right) = plt.subplots(1,2,figsize=(12.5,5.7),gridspec_kw={'width_ratios':[1,1.12]})
    fig.subplots_adjust(left=.073,right=.975,bottom=.235,top=.735,wspace=.30)
    fig.suptitle('Cross-company replacement: accuracy and response outcomes',
                 x=.073,y=.975,ha='left',fontsize=16,fontweight='bold')
    fig.text(.073,.914,'FinanceBench development set | 30 paired questions | Five blocks per input | Seed 2026',fontsize=10.5)
    blue,gray,amber,red = '#216B9A','#7B8998','#F0BE55','#BA4651'
    x = [r['noise_fraction']*100 for r in rows]
    values = [r['end_to_end_accuracy']*100 for r in rows]
    left.plot(x,values,color=blue,lw=2.5,marker='o',markersize=7)
    for pos,value,r in zip(x,values,rows):
        left.annotate(f'{value:.2f}%\n({r["correct_count"]}/{r["planned_count"]})',
                      (pos,value),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    left.set_title('A  End-to-end accuracy',loc='left',pad=16,fontweight='bold',fontsize=12)
    left.set(xlim=(-6,66),ylim=(0,100),xticks=x,yticks=[0,20,40,60,80,100],
             xlabel='Replaced blocks (%)',ylabel='Correct / planned cases')
    left.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    left.grid(axis='y',color='#E6EBF0',lw=.8)
    left.set_axisbelow(True)
    bottom = [0]*4
    categories = [('correct_count','Correct',blue,'white'),('wrong_answered_count','Incorrect answer',gray,'white'),
                  ('abstain_count','Abstention',amber,'#263344'),('schema_failure_count','Schema failure',red,'white')]
    for key,label,color,text_color in categories:
        heights = [r[key] for r in rows]
        right.bar(range(4),heights,bottom=bottom,width=.62,label=label,color=color,edgecolor='white',linewidth=.9)
        for i,h in enumerate(heights):
            if h:
                right.text(i,bottom[i]+h/2,str(h),ha='center',va='center',color=text_color,fontweight='bold')
        bottom = [a+b for a,b in zip(bottom,heights)]
    right.set_title('B  Outcome composition',loc='left',pad=16,fontweight='bold',fontsize=12)
    right.set(ylim=(0,30),yticks=[0,5,10,15,20,25,30],xticks=range(4),
              xticklabels=[f'{int(v)}%' for v in x],xlabel='Replaced blocks (%)',ylabel='Cases (n = 30 per condition)')
    right.grid(axis='y',color='#E6EBF0',lw=.8)
    right.set_axisbelow(True)
    right.legend(loc='lower left',bbox_to_anchor=(-.015,1.19),ncol=2,frameon=False,fontsize=9,columnspacing=1.1)
    fig.text(.073,.112,'Schema failures remain in end-to-end accuracy and are not counted as valid abstentions.',fontsize=10)
    fig.text(.073,.065,'Replacement both removes original evidence and inserts distractor candidates. One development set and one seed; descriptive results only.',fontsize=9,color='#58677A')
    fig.savefig(path/'noise_main_results.png',dpi=220,facecolor='white')
    fig.savefig(path/'noise_main_results.svg',facecolor='white')
    plt.close(fig)


def percent(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def report(rows,clean,paired,source):
    lines = ['# Cross-company replacement: development-set results','',
        '## Experimental scope','',
        'The frozen experiment evaluates the same 30 FinanceBench development questions at 0%, 20%, 40% and 60% block replacement. Each input contains five evidence blocks, with zero to three blocks replaced by cross-company distractor candidates. The random seed is 2026. The 0% condition is this formal run\'s own baseline, not an earlier Dense RAG run and not a guarantee of sufficient evidence.','',
        f'Model: `{source["sources"]["model"]}`. Prompt: `{source["sources"]["prompt_version"]}`. Source run: `{source["sources"]["run_id"]}`.','',
        '## Results','',
        '| Replaced blocks | Correct / all | End-to-end accuracy | Clean accuracy | Abstentions / all | Schema failures / all | Citation support / applicable |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r,c in zip(rows,clean):
        n = r['planned_count']
        lines.append(f'| {r["noise_fraction"]:.0%} | {r["correct_count"]}/{n} | {percent(r["end_to_end_accuracy"])} | {c["correct_count"]}/{c["planned_count"]} = {percent(c["end_to_end_accuracy"])} | {r["abstain_count"]}/{n} | {r["schema_failure_count"]}/{n} | {r["citation_supported_count"]}/{r["citation_applicable_count"]} = {percent(r["citation_support_accuracy_applicable"])} |')
    delta = 100*(rows[-1]['end_to_end_accuracy']-rows[0]['end_to_end_accuracy'])
    lines.extend(['','## Interpretation','',
        f'End-to-end accuracy was {percent(rows[0]["end_to_end_accuracy"])} at 0% replacement and {percent(rows[-1]["end_to_end_accuracy"])} at 60%, a change of {delta:+.2f} percentage points. The pattern was not monotonic: the 20% condition had {rows[1]["correct_count"]} correct answers, versus {rows[0]["correct_count"]} at 0%. A one-question difference in this small development set does not establish a general benefit from adding noise.','',
        f'At 60%, {rows[-1]["abstain_count"]} cases were valid abstentions and {rows[-1]["schema_failure_count"]} were schema failures. These are distinct outcomes. Among the {rows[-1]["valid_output_count"]} schema-valid outputs, accuracy was {percent(rows[-1]["valid_output_accuracy"])}; this conditional rate must not replace the end-to-end rate of {percent(rows[-1]["end_to_end_accuracy"])}.','',
        'Aggregate accuracy can hide changes in which questions are answered correctly. The following comparisons pair each question with its own 0% output. "Incorrect" here includes wrong or incomplete answers, valid abstentions and schema failures.','',
        '| Comparison with 0% | Both correct | Correct → incorrect | Incorrect → correct | Both incorrect |',
        '|---|---:|---:|---:|---:|'])
    for p in paired:
        lines.append(f'| {p["noise_fraction"]:.0%} | {p["both_correct"]} | {p["correct_to_incorrect"]} | {p["incorrect_to_correct"]} | {p["both_incorrect"]} |')
    lines.extend(['','## Metric definitions and limits','',
        '- End-to-end accuracy includes all 30 cases per condition; schema failures score zero.',
        '- Clean accuracy excludes only `financebench_id_00283`, consistently at every level; other disputed labels and notes are retained.',
        '- Citation support uses nonblank, applicable labels, including substantive claims made in some abstentions. It is conditional on a changing set of applicable cases, not a common-denominator causal comparison. Schema failures and claim-free abstentions have no support label.',
        '- Source hallucination means invented or falsely attributed source content. Insufficient citation support, arithmetic mistakes and explicitly disclosed assumptions alone do not qualify. Historical labels need the same rubric before their rates can be compared.',
        '- Page-level gold hits are diagnostic only and do not establish that a retrieved chunk contains sufficient evidence.',
        '- Replacement simultaneously removes original evidence and adds distractor candidates. This design does not isolate pure distraction.',
        '- These results concern 30 development questions and one seed. Repeated conditions are paired, not 120 independent questions. No statistical-significance or broad-generalization claim is made.',
        '- Labels are the confirmed annotations recorded in the supplied workbook. The plotting step neither rejudges them nor identifies the reviewer.','',
        '## Figure caption','',
        '**Figure 1.** End-to-end answer accuracy and outcome composition under cross-company block replacement. Each condition contains the same 30 questions. Panel A reports accuracy over all planned cases; panel B separates correct answers, incorrect answers, valid abstentions and schema failures. Counts are shown inside bars. Connecting line segments are visual guides, not a fitted response model. No uncertainty intervals are estimated.','',
        '## Reproducibility','',
        'Generated by `plot_noise_results.py` from a completed `calculate_noise_metrics.py` output directory. The plotting script checks the metrics-file hashes, condition totals and paired counts before writing a new directory. It does not call a model API or modify the source workbook, raw run or previous metrics.',''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-dir',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=Path('results/figures/noise_formal'))
    args = parser.parse_args(argv)
    data,manifest,hashes = load(args.metrics_dir)
    rows,clean,paired = validate(data)
    # Check plotting dependency before creating the output directory.
    import matplotlib
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output_root/('run_'+stamp+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    draw(rows,output)
    with (output/'noise_results_en.md').open('x',encoding='utf-8') as f:
        f.write(report(rows,clean,paired,manifest))
    _,_,current_hashes = load(args.metrics_dir)
    require(current_hashes==hashes,'Metrics files changed during plotting; export is incomplete')
    provenance = {'complete':True,'plot_version':'noise_plot_v1.0','metrics_directory':str(args.metrics_dir.resolve()),
                  'source_run_id':manifest['sources']['run_id'],'input_sha256':hashes,'matplotlib_version':matplotlib.__version__,
                  'script_sha256':sha(Path(__file__).read_bytes()),
                  'output_sha256':{p.name:sha(p.read_bytes()) for p in sorted(output.iterdir())}}
    with (output/'figures_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(provenance,f,indent=2,ensure_ascii=False)
        f.write('\n')
    print('Validated metric hashes, four condition totals and paired comparisons.')
    print('Saved English PNG, SVG, results note and figures_manifest.json.')
    print('Output: '+str(output.resolve()))
    print('No API calls; source inputs and earlier outputs unchanged.')
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except ImportError as exc:
        print('Missing dependency. In your active environment run: python -m pip install matplotlib',file=sys.stderr)
        raise SystemExit(1) from exc
    except (ValueError,KeyError,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        raise SystemExit(1) from exc
