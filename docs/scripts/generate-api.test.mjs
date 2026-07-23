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

test('convertSphinxMarkup escapes dollar signs inside math roles', () => {
  assert.equal(
    convertSphinxMarkup('sum :math:`x + $y$` and :math:`z`'),
    'sum $x + \\$y\\$$ and $z$',
  );
});

test('convertSphinxMarkup restores escaped braces inside math roles', () => {
  assert.equal(
    convertSphinxMarkup('phase :math:`i^\\{\\binom\\{w\\}\\{2\\}\\}`'),
    'phase $i^{\\binom{w}{2}}$',
  );
  assert.equal(
    convertSphinxMarkup('term :math:`m_\\{i_1\\} \\cdots m_\\{i_w\\}`'),
    'term $m_{i_1} \\cdots m_{i_w}$',
  );
});
