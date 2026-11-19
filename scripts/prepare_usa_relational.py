"""Prepare a fixed-2010-geography, retrospective USA exploratory dataset."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from usa_raw_inventory import inspect_raw

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def prepare(raw, target, audit):
 raw,target,audit=map(Path,(raw,target,audit))
 if (target/'provenance.json').exists():raise FileExistsError('Prepared dataset already exists; use a new target to preserve provenance.')
 target.mkdir(parents=True,exist_ok=True)
 inspect_raw(raw,audit)
 gaz=pd.read_csv(raw/'Gaz_counties_national.txt',sep='\t',encoding='cp1252',dtype={'GEOID':str})
 gaz.columns=gaz.columns.str.strip()
 # 50 states and DC, excluding Puerto Rico/territories; valid county registry from Gazetteer.
 gaz=gaz[gaz.GEOID.str[:2].astype(int)<=56].copy()
 nodes=gaz.rename(columns={'GEOID':'fips','POP10':'population','INTPTLAT':'latitude','INTPTLONG':'longitude','NAME':'name'})[['fips','name','population','latitude','longitude']].sort_values('fips')
 if nodes.fips.duplicated().any() or (nodes.population<=0).any():raise ValueError('Invalid Gazetteer nodes')
 # Keep counties without SCI: missing relational shifts are zero, not grounds for dropping origins.
 nodes.to_csv(target/'nodes.csv',index=False);codes=set(nodes.fips)
 flows=pd.read_csv(audit/'irs_candidate_routes_NOT_HARMONIZED.csv',dtype={'origin':str,'destination':str})
 keys=['period','origin','destination']
 if (flows.groupby(keys)['count'].nunique()>1).any():raise ValueError('Conflicting IRS duplicate counts')
 duplicated=flows.duplicated(keys);dedup_count=int(duplicated.sum());flows=flows[~duplicated].copy()
 retained=flows.origin.isin(codes)&flows.destination.isin(codes)
 dropped=flows[~retained]
 dropped.to_csv(audit/'irs_excluded_routes_geography.csv',index=False)
 summary=[]
 for year,g in flows.groupby('period'):
  keep=g.origin.isin(codes)&g.destination.isin(codes)
  summary.append(dict(period=year,retained_count=g.loc[keep,'count'].sum(),excluded_count=g.loc[~keep,'count'].sum(),excluded_routes=int((~keep).sum())))
 pd.DataFrame(summary).to_csv(audit/'flow_support_audit.csv',index=False)
 flows[retained].to_csv(target/'flows.csv',index=False)
 idx={c:i for i,c in enumerate(nodes.fips)};n=len(nodes)
 seen=np.zeros((n,n),dtype=bool);sci_missing_rows=0;first=True;outside=set()
 for chunk in pd.read_csv(raw/'us_counties.csv',dtype={'user_region':str,'friend_region':str},chunksize=300000):
  chunk.user_region=chunk.user_region.str.zfill(5);chunk.friend_region=chunk.friend_region.str.zfill(5)
  outside.update((set(chunk.user_region)|set(chunk.friend_region))-codes)
  chunk=chunk[chunk.user_region.isin(codes)&chunk.friend_region.isin(codes)&chunk.user_region.ne(chunk.friend_region)].copy()
  if chunk.duplicated(['user_region','friend_region']).any():raise ValueError('Duplicated SCI pairs')
  i=chunk.user_region.map(idx).to_numpy();j=chunk.friend_region.map(idx).to_numpy()
  if seen[i,j].any():raise ValueError('Duplicated SCI across chunks')
  val=pd.to_numeric(chunk.scaled_sci,errors='raise').to_numpy(float)
  if (val<0).any() or np.isinf(val).any():raise ValueError('Invalid SCI value')
  seen[i,j]=np.isfinite(val);sci_missing_rows+=int(np.isnan(val).sum())
  frame=chunk.rename(columns={'user_region':'origin','friend_region':'destination'})[['origin','destination','scaled_sci']]
  frame.to_csv(target/'sci.csv',mode='w' if first else 'a',header=first,index=False);first=False
 off=~np.eye(n,dtype=bool)
 rows=pd.DataFrame({'fips':nodes.fips.to_numpy(),'observed_sci_destinations':seen.sum(axis=1)})
 rows.to_csv(audit/'sci_coverage_by_origin.csv',index=False)
 sources=[raw/'Gaz_counties_national.txt',raw/'us_counties.csv',*sorted(raw.glob('*/countyoutflow*.csv'))]
 meta={'status':'exploratory retrospective; SCI vintage not supplied',
       'population':'POP10 from 2010 National Counties Gazetteer, fixed across periods',
       'coordinates':'2010 Gazetteer representative internal points; great-circle distance',
       'geography':'50 states + DC on fixed 2010 county codes; no inferred territorial crosswalk; out-of-support IRS corridors excluded and audited',
       'unreported_flow_policy':'recorded_counts_only','flow_field':'IRS n2 individuals/exemptions; nonnegative published individual-county routes only',
       'origin_totals':'sum of retained reported corridor counts, not IRS aggregate outflow',
       'sci_release':None,'sci_missing_policy':'zero relational shift; retain nodes with no SCI',
       'counties':n,'sci_missing_offdiagonal':int((~seen&off).sum()),'sci_missing_fraction':float((~seen&off).sum()/off.sum()),
       'no_sci_origins':nodes.loc[seen.sum(axis=1)==0,'fips'].tolist(),
       'irs_identical_duplicate_rows_removed':dedup_count,
       'irs_excluded_codes':sorted((set(dropped.origin)|set(dropped.destination))-codes),
       'sci_codes_outside_geography':sorted(outside),
       'periods':sorted(flows.period.unique()),'source_sha256':{str(p.resolve()):sha(p) for p in sources}}
 (target/'provenance.json').write_text(json.dumps(meta,indent=2))
 return meta

if __name__=='__main__':
 root=Path(__file__).resolve().parents[1]
 print(json.dumps(prepare(root.parent/'FERM_python/data/us',root/'data/usa_relational/prepared',root/'outputs/usa_relational/input_audit'),indent=2))
