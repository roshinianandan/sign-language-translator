/* Rule-based gesture classifier - mirror of rules.py (keep in sync). */
const GESTURE_GUIDE = [
  { key: 'hello',    how: 'Open palm - all 5 fingers up' },
  { key: 'yes',      how: 'Thumbs up (fist, thumb up)' },
  { key: 'no',       how: 'Thumbs down (fist, thumb down)' },
  { key: 'iloveyou', how: 'Thumb + index + little finger out' },
  { key: 'thanks',   how: 'Peace sign (index + middle finger)' },
  { key: 'please',   how: 'Point up with index finger only' },
  { key: 'help',     how: 'Closed fist (thumb tucked in)' },
];
const FINGER_PAIRS = [[6, 8], [10, 12], [14, 16], [18, 20]];
const FINGER_RATIO = 1.2, THUMB_RATIO = 0.65;

function _d(a, b, w, h) { return Math.hypot((a.x - b.x) * w, (a.y - b.y) * h); }

function classifyGesture(lm, w, h) {
  const wrist = lm[0];
  const [i, m, r, p] = FINGER_PAIRS.map(([pip, tip]) => _d(lm[tip], wrist, w, h) > FINGER_RATIO * _d(lm[pip], wrist, w, h));
  const palm = _d(wrist, lm[9], w, h) + 1e-6;
  const thumb = _d(lm[4], lm[5], w, h) / palm > THUMB_RATIO && _d(lm[4], lm[17], w, h) > _d(lm[3], lm[17], w, h);
  if (i && m && r && p) return 'hello';
  if (thumb && i && p && !m && !r) return 'iloveyou';
  if (i && m && !r && !p) return 'thanks';
  if (i && !(m || r || p)) return 'please';
  if (!(i || m || r || p)) {
    if (thumb) {
      const dx = (lm[4].x - lm[2].x) * w, dy = (lm[4].y - lm[2].y) * h;
      if (Math.abs(dy) > 1.2 * Math.abs(dx)) return dy < 0 ? 'yes' : 'no';
      return null;
    }
    return 'help';
  }
  return null;
}
if (typeof module !== 'undefined') module.exports = { classifyGesture, GESTURE_GUIDE };
