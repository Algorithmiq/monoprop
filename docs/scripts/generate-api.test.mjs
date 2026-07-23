import assert from 'node:assert/strict';
import test from 'node:test';

import { convertSphinxMarkup } from './generate-api.mjs';

test('convertSphinxMarkup renders Sphinx math as markdown math', () => {
  assert.equal(
    convertSphinxMarkup(
      'phase :math:`i^{\\binom{w}{2}}` and :class:`~monoprop.circuit.ExpGate`',
    ),
    'phase $i^{\\binom{w}{2}}$ and `monoprop.circuit.ExpGate`',
  );
});
