import { readFileSync } from 'node:fs';
import * as r from '../src/game/rules.js';
const states=JSON.parse(readFileSync(0,'utf8'));
console.log(JSON.stringify(states.map(s=>({dest:r.legalDestinations(s), distances:[r.distance(s,1),r.distance(s,2)], errors:Array.from({length:209},(_,i)=>r.actionError(s,i)),mask:r.actionMask(s),obs:r.observation(s), next:r.legalActions(s).slice(0,4).map(a=>r.applyAction(s,a))}))));
